"""MySQL Checkpointer 示例：使用 PyMySQLSaver 持久化图状态。

需要安装：uv pip install "langgraph-checkpoint-mysql[pymysql]"
MySQL 连接信息通过环境变量或直接传参配置。

前置步骤（建库建表）：
    1. 创建数据库: CREATE DATABASE IF NOT EXISTS langgraph;
    2. 调用 PyMySQLSaver.setup() 自动建表（下方 4 张表）。

═══════════════════════════════════════════════════════════════════════════════
表结构说明（4 张表，均由 setup() 自动创建）
═══════════════════════════════════════════════════════════════════════════════

1. checkpoints — 核心表，每次图执行一个 super-step 就写入一行，记录该步结束后图的快照元信息
   ┌───────────────────────┬─────────────┬──────────────────────────────────────┐
   │ 字段                   │ 类型         │ 说明                                  │
   ├───────────────────────┼─────────────┼──────────────────────────────────────┤
   │ thread_id             │ varchar(150) │ 会话/线程 ID，区分不同对话              │
   │ checkpoint_ns         │ varchar(2000)│ 命名空间: "" = 父图, "子图:uuid" = 子图  │
   │ checkpoint_id         │ varchar(150) │ 当前 checkpoint 的唯一 ID (UUID)        │
   │ parent_checkpoint_id  │ varchar(150) │ 上一个 checkpoint ID，形成链表可回溯历史 │
   │ checkpoint            │ json         │ channel_versions 映射，记录每个 channel │
   │                       │             │ 当前的版本号，如 {"foo":"v2","bar":"v3"} │
   │ metadata              │ json         │ 步骤元信息，如 {"step":0,"source":"loop"}│
   └───────────────────────┴─────────────┴──────────────────────────────────────┘

2. checkpoint_blobs — 数据表，存储每个 channel 的实际序列化数据
   checkpoints 表只存版本号，真正的状态值（foo="a"、bar=["a","b"]）存在这里
   ┌───────────────────────┬─────────────┬──────────────────────────────────────┐
   │ 字段                   │ 类型         │ 说明                                  │
   ├───────────────────────┼─────────────┼──────────────────────────────────────┤
   │ thread_id             │ varchar(150) │ 会话/线程 ID                           │
   │ channel               │ varchar(150) │ channel 名 = State 字段名              │
   │                       │             │ (如 "foo"、"bar"、"__start__")           │
   │ version               │ varchar(150) │ 该 channel 版本号，与 checkpoints 表    │
   │                       │             │ 的 channel_versions 对应                 │
   │ type                  │ varchar(150) │ 序列化方式: msgpack / json / pickle      │
   │ blob                  │ longblob     │ 序列化后的二进制数据                      │
   └───────────────────────┴─────────────┴──────────────────────────────────────┘
   举例: state = {"foo": "a", "bar": ["a","b"]} 会生成 2 条记录:
     channel="foo", blob=<序列化后的 "a">
     channel="bar", blob=<序列化后的 ["a","b"]>

3. checkpoint_writes — 暂存表，记录一个 super-step 内每个 task(节点)的写入操作
   用于支持部分写入和中断恢复: step 执行到一半崩溃，可通过此表恢复
   ┌───────────────────────┬─────────────┬──────────────────────────────────────┐
   │ 字段                   │ 类型         │ 说明                                  │
   ├───────────────────────┼─────────────┼──────────────────────────────────────┤
   │ thread_id             │ varchar(150) │ 会话/线程 ID                           │
   │ checkpoint_id         │ varchar(150) │ 关联的 checkpoint ID                    │
   │ task_id               │ varchar(150) │ 执行写入的 task（节点执行）ID             │
   │ idx                   │ int          │ 同一 task 内多次写入的序号               │
   │ channel               │ varchar(150) │ 写入的目标 channel 名                   │
   │ type                  │ varchar(150) │ 序列化方式                               │
   │ blob                  │ longblob     │ 本次写入的数据                           │
   │ task_path             │ varchar(2000)│ task 在图中的路径                        │
   └───────────────────────┴─────────────┴──────────────────────────────────────┘

4. checkpoint_migrations — 迁移表，记录已应用的 schema 迁移版本号
   只有一列 v (int)，每行一个版本号。setup() 执行时自动检查并应用增量迁移。

关系:
    checkpoints (元信息 + channel_versions 版本号)
        │
        ├── checkpoint_blobs (实际数据，通过 thread_id + channel + version 关联)
        │
        └── checkpoint_writes (暂存写入，通过 thread_id + checkpoint_id 关联)
"""

import os
from operator import add
from typing import Annotated, Any

import pymysql
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


# ── 连接配置 ──────────────────────────────────────────────────────────────────

MYSQL_CONFIG = {
    "host": os.environ.get("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.environ.get("MYSQL_PORT", "3306")),
    "user": os.environ.get("MYSQL_USER", "root"),
    "password": os.environ.get("MYSQL_PASSWORD", "root"),
    "database": os.environ.get("MYSQL_DATABASE", "langgraph"),
    "autocommit": True,
}


# ── Graph 状态 ────────────────────────────────────────────────────────────────

class MySQLState(TypedDict):
    foo: str
    bar: Annotated[list[str], add]


# ── 建库建表 ────────────────────────────────────────────────────────────────────

def setup_database() -> None:
    """确保 langgraph 数据库和所需表已存在（幂等操作）。"""
    # 先连默认库，建 database
    conn = pymysql.connect(
        host=MYSQL_CONFIG["host"],
        port=MYSQL_CONFIG["port"],
        user=MYSQL_CONFIG["user"],
        password=MYSQL_CONFIG["password"],
        autocommit=True,
    )
    conn.cursor().execute(
        f"CREATE DATABASE IF NOT EXISTS `{MYSQL_CONFIG['database']}` "
        "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci"
    )
    conn.close()

    # 再连目标库，检查表是否已存在，避免 setup() 重复建索引报错
    conn = pymysql.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES "
        "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'checkpoint_migrations'",
        (MYSQL_CONFIG["database"],),
    )
    tables_exist = cursor.fetchone()[0] > 0

    if not tables_exist:
        saver = PyMySQLSaver(conn)
        saver.setup()
        print(f"✓ 数据库 `{MYSQL_CONFIG['database']}` 及表已创建")
    else:
        print(f"✓ 数据库 `{MYSQL_CONFIG['database']}` 及表已存在，跳过建表")

    conn.close()


# ── 构建图 ─────────────────────────────────────────────────────────────────────

def _build_graph(checkpointer: PyMySQLSaver):
    def node_a(state: MySQLState) -> dict[str, Any]:
        return {"foo": "a", "bar": ["a"]}

    def node_b(state: MySQLState) -> dict[str, Any]:
        return {"foo": "b", "bar": ["b"]}

    workflow = StateGraph(MySQLState)
    workflow.add_node("node_a", node_a)
    workflow.add_node("node_b", node_b)
    workflow.add_edge(START, "node_a")
    workflow.add_edge("node_a", "node_b")
    workflow.add_edge("node_b", END)
    return workflow.compile(checkpointer=checkpointer)


# ── 演示 ──────────────────────────────────────────────────────────────────────

class MySQLCheckpointerDemo:
    """演示 PyMySQLSaver 持久化 + 恢复图状态。"""

    def run(self, thread_id: str = "mysql-demo-1") -> None:
        conn = pymysql.connect(**MYSQL_CONFIG)
        checkpointer = PyMySQLSaver(conn)

        graph = _build_graph(checkpointer)
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # 第一次运行
        graph.invoke({"foo": "", "bar": []}, config)
        state = graph.get_state(config)
        print(f"[首次运行] thread={thread_id}  state={state.values}")

        # 再次运行（同一 thread_id），bar 会累加
        graph.invoke({"foo": "", "bar": ["x"]}, config)
        state = graph.get_state(config)
        print(f"[二次运行] thread={thread_id}  state={state.values}")

        # 查看历史 checkpoint
        history = list(graph.get_state_history(config))
        print(f"[历史] 共 {len(history)} 个 checkpoint")

        # 列出所有 thread
        threads = list(checkpointer.list({}))
        print(f"[所有 thread] 共 {len(threads)} 个")

        conn.close()


class TimeTravelDemo:
    """演示从历史 checkpoint 回放（时间旅行）。

    流程：
      1. 正常执行图，产生多个 checkpoint
      2. 查看历史，找到某个中间 checkpoint
      3. 从该 checkpoint 重新 invoke，图会从那个状态继续执行
    """

    def run(self, thread_id: str = "time-travel-demo") -> None:
        conn = pymysql.connect(**MYSQL_CONFIG)
        checkpointer = PyMySQLSaver(conn)
        graph = _build_graph(checkpointer)
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # ── 第 1 步：正常执行，产生 checkpoint 链 ──
        graph.invoke({"foo": "", "bar": []}, config)
        graph.invoke({"foo": "", "bar": ["x"]}, config)
        print(f"[正常执行] 最终状态: {graph.get_state(config).values}")

        # ── 第 2 步：查看历史 checkpoint ──
        history = list(graph.get_state_history(config))
        print(f"\n[历史] 共 {len(history)} 个 checkpoint（从新到旧）:")
        for i, h in enumerate(history):
            cp_id = h.config["configurable"]["checkpoint_id"][:8]
            meta = h.metadata
            print(f"  [{i}] checkpoint_id={cp_id}...  values={h.values}  meta={meta}")

        # ── 第 3 步：从某个历史 checkpoint 回放 ──
        # 选择倒数第 2 个（即第一次 invoke 刚完成时的状态）
        target = history[2]
        replay_config = target.config
        print(f"\n[回放] 从 checkpoint {target.config['configurable']['checkpoint_id'][:8]}... 回放")
        print(f"  回放前状态: {target.values}")

        # 用该 checkpoint 的 config 重新 invoke，图会从这个历史状态继续
        graph.invoke({"foo": "replay", "bar": ["replayed"]}, replay_config)
        new_state = graph.get_state(config)
        print(f"  回放后状态: {new_state.values}")
        print(f"  （注意 bar 是在回放点的值基础上累加，而非最终值）")

        # ── 对比：最终历史分叉 ──
        new_history = list(graph.get_state_history(config))
        print(f"\n[分叉] 回放后历史共 {len(new_history)} 个 checkpoint")
        for i, h in enumerate(new_history):
            cp_id = h.config["configurable"]["checkpoint_id"][:8]
            print(f"  [{i}] checkpoint_id={cp_id}...  foo={h.values.get('foo')}")

        conn.close()


class MySQLFromConnStringDemo:
    """演示通过连接字符串创建 PyMySQLSaver（上下文管理器方式）。"""

    def run(self) -> None:
        conn_string = (
            f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}"
            f"@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}"
            f"/{MYSQL_CONFIG['database']}"
        )
        print(f"连接字符串: {conn_string}")

        with PyMySQLSaver.from_conn_string(conn_string) as checkpointer:
            graph = _build_graph(checkpointer)
            config: RunnableConfig = {"configurable": {"thread_id": "conn-string-demo"}}

            graph.invoke({"foo": "", "bar": []}, config)
            print(f"from_conn_string 运行成功: {graph.get_state(config).values}")


# ── 清空数据 ──────────────────────────────────────────────────────────────────

def cleanup_tables() -> None:
    """清空 4 张表的数据（保留表结构），方便重新运行演示。"""
    conn = pymysql.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()
    # 按依赖顺序删除：先删子表，再删主表
    cursor.execute("DELETE FROM checkpoint_writes")
    cursor.execute("DELETE FROM checkpoint_blobs")
    cursor.execute("DELETE FROM checkpoints")
    cursor.execute("DELETE FROM checkpoint_migrations")
    conn.commit()
    conn.close()
    print("✓ 4 张表数据已清空")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 建库建表 ===")
    setup_database()

    print("\n=== 清空历史数据 ===")
    cleanup_tables()

    print("\n=== MySQL Checkpointer 基本演示 ===")
    MySQLCheckpointerDemo().run()

    print("\n=== from_conn_string 方式 ===")
    MySQLFromConnStringDemo().run()

    print("\n=== 时间旅行（从历史 checkpoint 回放） ===")
    TimeTravelDemo().run()
