"""
LLM客户端工厂模块
统一使用LangChain的ChatOpenAI作为客户端。
通义千问(Qwen)、深度求索(DeepSeek)、OpenAI 三者都兼容OpenAI接口协议，
因此一套客户端代码即可适配三种模型后端。
切换模型只需要修改.env环境变量：
    LLM_BASE_URL=...
    LLM_API_KEY=...
    LLM_MODEL=...

设计说明：刻意不缓存构建好的客户端实例。
ChatOpenAI本质只是一个轻量配置容器，每次调用都新建实例，可以方便单元测试时做monkeypatch打桩。
真正做缓存的是get_settings()配置读取函数。
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 读取项目配置
from app.config import get_settings


def get_chat_model() -> ChatOpenAI:
    """
    根据.env配置返回全新的ChatOpenAI对话模型实例。

    第1阶段：仅在chat.py中调用本函数。
    第7阶段（工具调用能力）：SQL生成节点也会调用该函数获取模型。
    :return: 配置完成的ChatOpenAI实例
    """
    settings = get_settings()
    return ChatOpenAI(
        base_url=settings.llm_base_url,       # LLM接口地址
        api_key=settings.llm_api_key,         # API密钥
        model=settings.llm_model,             # 模型名称
        temperature=0.1,                      # 低温度，SQL生成需要结果尽量确定、少随机性
        max_tokens=2000,                       # 最大输出token数
        timeout=30,                            # 请求超时时间，单位秒
    )


def get_embeddings() -> OpenAIEmbeddings:
    """
    返回兼容OpenAI协议的Embeddings向量化客户端，第5阶段RAG会使用。

    第1阶段就把该函数放在这里，提前固化导入依赖关系；
    后续业务代码写 `from app.llm import get_embeddings`，代码无需改动。
    :return: 向量化模型客户端
    """
    settings = get_settings()
    return OpenAIEmbeddings(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
    )