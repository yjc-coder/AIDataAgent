"""MySQL 异步连接池生命周期管理。

使用模块级全局连接池（单例），任何模块需要数据库连接时直接通过 get_pool() 获取，
不必把连接对象层层透传。连接池在 app.main 的 lifespan 启动钩子中创建，
在应用关闭钩子中释放。

为什么选 aiomysql？
    - 真正的异步驱动，与 FastAPI 的 async 请求处理天然契合，不会阻塞事件循环。
    - 第2阶段只做只读 SELECT 查询，连接池规模 1~10 完全够用。

为什么用单例连接池？
    - 复用 TCP 连接和认证握手，比每个请求新建连接便宜得多。
    - 让 execute.py 和 repository.py 共享同一个池对象，避免重复创建、连接数失控。
"""
from __future__ import annotations

from typing import Optional

import aiomysql

from app.config import get_settings


# 模块级全局连接池实例。初始为 None，表示尚未初始化；
# 由 init_pool() 赋值，close_pool() 重新置空。
_pool: Optional[aiomysql.Pool] = None


async def init_pool() -> aiomysql.Pool:
    """创建全局连接池（幂等）。

    重复调用安全：如果连接池已经存在，直接返回现有实例，不会重复创建。
    所有连接参数（主机、端口、账号、库名、字符集、池大小）均从 .env 配置读取。
    """
    global _pool
    # 幂等保护：已初始化则直接复用
    if _pool is not None:
        return _pool
    s = get_settings()
    _pool = await aiomysql.create_pool(
        host=s.db_host,          # MySQL 主机地址
        port=s.db_port,          # MySQL 端口，默认 3306
        user=s.db_user,          # 数据库用户名
        password=s.db_password,  # 数据库密码
        db=s.db_name,            # 默认连接的数据库名
        charset=s.db_charset,    # 字符集，通常为 utf8mb4
        autocommit=True,         # 第2阶段：只读链路，不需要显式事务，自动提交即可
        minsize=s.db_pool_min,   # 连接池保有的最小空闲连接数
        maxsize=s.db_pool_max,   # 连接池允许的最大连接数（并发上限）
    )
    return _pool


async def close_pool() -> None:
    """关闭并清空全局连接池（幂等）。

    重复调用安全：连接池不存在时直接返回。
    close() 发起关闭，wait_closed() 等待池内连接真正释放完毕。
    """
    global _pool
    if _pool is None:
        return
    _pool.close()              # 停止接受新连接，开始回收现有连接
    await _pool.wait_closed()  # 异步等待所有连接关闭完成
    _pool = None               # 置空，便于下次重新初始化（也方便单元测试）


def get_pool() -> aiomysql.Pool:
    """获取当前全局连接池；若启动钩子尚未执行则抛出 RuntimeError。

    这是一个同步函数：连接池必须已在 lifespan 启动阶段通过 init_pool() 创建好。
    此处快速失败（fail-fast），把"忘了初始化"的配置错误尽早暴露出来。
    """
    if _pool is None:
        raise RuntimeError(
            "数据库连接池尚未初始化。app.main 的 lifespan 启动钩子是否执行了？"
        )
    return _pool
