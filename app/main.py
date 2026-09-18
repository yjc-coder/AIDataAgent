"""
FastAPI 应用入口文件

对外暴露接口清单：
    GET  /health            — 存活探针（健康检查）
    POST /api/chat          — 单轮原始LLM调用（阶段1）
    POST /api/sql           — 基于LangGraph的Text-to-SQL主流程（阶段6）
    POST /api/chat/stream   — SSE流式节点进度推送（阶段9）
    GET  /api/graph/*       — 图谱结构与调用链路追踪接口
    GET  /                   — Vue3聊天前端页面（阶段10）
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# 导入各个模块的路由
from app.api.chat import router as chat_router
from app.api.graph import router as graph_router
from app.api.sql import router as sql_router
from app.api.stream import router as stream_router
# 初始化Agent全局辅助对象
from app.agent.graph import init_default_helpers
# 数据库连接池启动/关闭
from app.database.engine import close_pool, init_pool
# 健康接口返回模型
from app.models import HealthResponse


# 在模块导入时一次性加载生产环境Agent助手实例
# 如果单元测试需要打桩(mock)，要在第一次调用get_compiled_graph()之前执行 set_helpers(...)
init_default_helpers()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期钩子：应用启动前、关闭后执行回调（替代旧版 startup / shutdown）
    阶段2：服务启动时创建MySQL连接池；服务关闭时销毁连接池

    设计说明：
    init_pool 如果失败，**不会直接让服务崩溃退出**。
    健康探针 /health 依然可以正常响应，运维人员可以进入服务排查数据库问题。
    """
    try:
        await init_pool()
    except Exception as e:  # pragma: no cover：单元测试不会覆盖这个异常分支
        import logging
        logging.getLogger(__name__).warning("数据库连接池初始化失败: %s", e)
    # yield 之前：应用启动阶段；yield 之后：应用停止阶段
    yield
    # 服务关闭，释放数据库连接池
    await close_pool()


# 创建FastAPI应用实例
app = FastAPI(
    title="AI Data Agent",
    description=(
        "轻量级Text-to-SQL智能数据查询Agent，用于作品集项目。 "
        "使用FastAPI + LangChain + LangGraph实现，替代Spring AI Alibaba DataAgent方案。"
    ),
    version="1.0.0",
    lifespan=lifespan, # 绑定生命周期钩子
)

# CORS跨域中间件：本地开发环境放开全部跨域权限
# 生产环境必须改成通过环境变量配置允许的域名，不能用 ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """存活探针：不调用LLM、不访问数据库，轻量接口，供K8s/运维做服务保活检测"""
    return HealthResponse()


# 注册各个业务路由，统一前缀 /api
app.include_router(chat_router, prefix="/api", tags=["chat"])
app.include_router(sql_router, prefix="/api", tags=["sql"])
app.include_router(stream_router, prefix="/api", tags=["stream"])
app.include_router(graph_router, prefix="/api", tags=["graph"])

# 挂载Vue3前端静态资源。放在路由注册最后！
# 原因：优先匹配上面的 /api 接口；剩下所有未匹配的路由交给前端静态页面（SPA单页应用路由）
_FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")