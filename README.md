# AI Data Agent

> 面向业务数据库的 AI Agent 数据分析系统：**Python + FastAPI + LangChain + LangGraph + RAG + Tool Calling + Text-to-SQL + SSE**

## 核心流程

```
用户问题
  ↓
IntentNode            判定 DATA_QUERY / CHAT（启发式 + LLM 结构化输出兜底）
  ↓
SchemaRecallNode      RAG：Chroma 召回 schema / 业务规则
  ↓  ↓（召回为空或无 schema 命中时）
ToolCallNode          LLM bind_tools 单轮决策 → 执行 search_schema / execute_sql
  ↓
SQLGenerateNode       LLM 结构化输出生成 SQL（带上一轮问答，支持"那2025年呢?"）
  ↓
SQLExecuteNode        sqlglot 校验 → 只读账号执行 MySQL
  ↓
ReportNode            LLM 将行数据总结为自然语言
  ↓
SSE 逐节点推送 → 前端时间线 + 最终答案
```

 LangGraph 两条条件边：`intent` 后 CHAT 直达 report；`schema_recall` 后低置信度绕行 `tool_call`。**LLM 只决定调什么工具，流程永远由图掌控——没有自主 Agent Loop。**

## 快速开始

### Docker（推荐）

```bash
cp .env.example .env      # 填入 LLM_API_KEY / EMBEDDING_API_KEY
docker compose up -d --build
# 打开 http://localhost:8065/
```

### 本地运行

```bash
docker compose up -d mysql          # 只起 MySQL
pip install -r requirements.txt
.venv\Scripts\python data\seed_data.py          # 灌种子数据（2000 订单 / 4958 明细）
.venv\Scripts\python -m uvicorn app.main:app --reload --port 8065
# 打开 http://127.0.0.1:8065/
```



```

