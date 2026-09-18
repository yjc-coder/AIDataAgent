基于 Python 的 AI 智能数据分析 Agent

目标技术栈：

text
Python 3.11+
FastAPI
LangChain
LangGraph
LLM API（Qwen / DeepSeek，优先使用 OpenAI-compatible API）
RAG
Chroma 或 FAISS
Tool Calling
Text-to-SQL
MySQL
Pydantic
SSE
Docker
Vue 3（仅保留简单前端）
最终项目应该能够作为我的第二个求职项目，与我的 Java 高并发项目形成互补：

text
Java 项目：
Spring Boot + MySQL + Redis + RabbitMQ + Redisson + Lua
→ Java 后端 / 高并发

AI 项目：
Python + FastAPI + LangGraph + RAG + Tool Calling + Text-to-SQL
→ AI Agent 应用开发
二、非常重要：项目定位
这个项目不是企业级 DataAgent 的完整复刻。

定位是：

轻量级、可理解、可运行、适合面试讲解的 AI Agent 应用。

核心目标不是堆技术，而是让我能够完整理解：

text
LLM
↓
Prompt
↓
RAG
↓
LangGraph
↓
Tool Calling
↓
Text-to-SQL
↓
MySQL
↓
SSE
因此必须遵守：

不要加入以下技术
text
Redis
RabbitMQ
Kafka
微服务
Kubernetes
Milvus
Elasticsearch
复杂 MCP
Multi-Agent
复杂 Planner
复杂 Agent Loop
复杂 Evaluation 平台
模型微调
本地大模型部署
CUDA
复杂消息队列
不要为了体现“高级”而增加不必要的技术。

三、改造原则
原则 1：先理解现有项目，再改造
不要直接删除当前项目。

第一步必须：

分析当前 Java 项目的目录结构

分析核心业务流程

找出现有：

StateGraph
Node
RAG
Schema
SQL 生成
SQL 执行
SSE
多轮对话
数据库
前端
建立 Java → Python 功能映射表

输出：

text
Java组件
↓
对应Python实现
↓
是否保留
↓
为什么
例如：

text
StateGraph
→ LangGraph

Spring AI ChatModel
→ LangChain ChatModel / OpenAI-compatible client

SimpleVectorStore
→ Chroma

NodeFunction
→ LangGraph Node

Controller
→ FastAPI Router

MyBatis
→ SQLAlchemy / SQLModel 或轻量数据库访问层

SSE
→ FastAPI StreamingResponse / EventSourceResponse
四、最终项目功能
最终至少实现以下功能。

1. Text-to-SQL
用户可以直接使用自然语言查询数据库。

例如：

text
查询2026年销售额最高的商品
系统自动：

text
理解问题
↓
检索Schema
↓
生成SQL
↓
SQL安全校验
↓
执行SQL
↓
分析结果
↓
返回自然语言答案
五、RAG
RAG 不用于普通文档问答。

本项目的 RAG 主要用于：

数据库 Schema 和业务知识检索。

知识库至少包含：

Schema知识
例如：

text
表名：orders

字段：
id：订单ID
user_id：用户ID
product_id：商品ID
amount：订单金额，单位为元
create_time：订单创建时间

业务说明：
该表用于记录用户订单信息。
业务知识
例如：

text
销售额 = 有效订单 amount 之和

退款订单不计入销售额

amount 单位为人民币元
当用户输入：

text
2026年哪个商品销售额最高？
RAG 应该召回：

text
orders
products
销售额业务定义
然后将检索结果提供给 SQL Generator。

六、LangGraph
使用 LangGraph 构建固定、可控的 Agent 工作流。

不要设计复杂的自主 Agent Loop。

最终采用：

text
START
 ↓
IntentNode
 ↓
SchemaRecallNode
 ↓
SQLGenerateNode
 ↓
SQLExecuteNode
 ↓
ReportNode
 ↓
END
每个节点职责明确。

七、Node 设计
1. IntentNode
职责：

判断用户请求是否属于数据分析请求。

例如：

text
查询2026年销售额
→ DATA_QUERY

查询销量最高商品
→ DATA_QUERY

你好
→ CHAT

帮我解释一下什么是SQL
→ CHAT
不要做复杂意图分类系统。

只需要满足当前项目即可。

2. SchemaRecallNode
职责：

调用 RAG。

流程：

text
用户问题
↓
Embedding
↓
Vector Store
↓
Similarity Search
↓
相关Schema + Business Knowledge
最终输出：

text
retrieved_schema
3. SQLGenerateNode
职责：

根据：

text
用户问题
+
RAG检索结果
+
对话上下文
调用 LLM 生成 SQL。

Prompt 必须明确：

text
你是MySQL专家。

只能生成SELECT语句。

不能执行INSERT、UPDATE、DELETE、DROP、ALTER等操作。

必须使用提供的Schema。

不能虚构不存在的表和字段。

默认限制结果数量。

只输出结构化SQL结果。
最好使用结构化输出，而不是依赖纯字符串解析。

八、Tool Calling
这是本项目相比原 Java 简化版本需要重点增加的能力。

只实现两个核心 Tool。

Tool 1：Schema Search Tool
定义：

python
search_schema(keyword)
功能：

根据关键词搜索相关：

text
表
字段
业务规则
Tool 2：SQL Execute Tool
定义：

python
execute_sql(sql)
但 Tool 内部不能直接执行 SQL。

必须：

text
execute_sql()
     ↓
SQL Validator
     ↓
只允许SELECT
     ↓
限制LIMIT
     ↓
执行MySQL
     ↓
返回结果
九、Tool Calling 的定位
不要实现复杂自主 Agent。

采用：

text
LangGraph
+
LLM
+
Tools
由 LangGraph 控制流程。

Agent 可以调用：

text
search_schema()
execute_sql()
但是系统必须对工具调用进行约束。

核心思想：

LLM 决定需要什么信息以及如何查询，程序负责控制工具权限和数据库安全。

十、SQL安全
必须实现基本 SQL 安全。

至少包括：

1. 只允许 SELECT
禁止：

text
INSERT
UPDATE
DELETE
DROP
ALTER
TRUNCATE
CREATE
REPLACE
2. 禁止危险函数
至少检测：

text
SLEEP
BENCHMARK
LOAD_FILE
3. LIMIT限制
默认：

text
LIMIT 1000
4. 数据库只读账号
项目文档中说明生产环境应使用：

text
只读数据库账号
只拥有 SELECT 权限。

5. SQL Parser
优先选择成熟 SQL Parser。

不要只使用简单字符串：

python
if "delete" in sql:
作为唯一安全机制。

可以采用：

text
SQL Parser
+
关键字检查
+
只读数据库账号
形成多层保护。

十一、多轮对话
支持简单的多轮上下文。

例如：

第一轮：

text
查询2026年的销售额
第二轮：

text
那2025年呢？
系统应该理解：

text
“那” = 销售额
State 至少包含：

text
messages
user_query
intent
retrieved_schema
generated_sql
query_result
final_answer
不要设计复杂 Memory 架构。

十二、SSE
提供：

text
POST /api/chat/stream
使用 SSE / Streaming Response。

用户能够看到类似：

text
正在理解问题...
正在检索相关数据表...
正在生成SQL...
正在执行查询...
正在分析结果...
最终：

text
2026年销售额最高的商品是 XXX，销售额为 XXX 元。
十三、FastAPI API设计
至少提供：

text
POST /api/chat
POST /api/chat/stream
GET  /api/health
其中：

/api/chat
普通请求：

json
{
    "query": "查询2026年销售额最高的商品",
    "conversation_id": "xxx"
}
返回：

json
{
    "answer": "...",
    "sql": "...",
    "data": [...]
}
/api/chat/stream
SSE：

text
event: thinking
data: 正在分析问题

event: retrieval
data: 正在检索Schema

event: sql
data: 正在生成SQL

event: result
data: 查询完成

event: answer
data: 最终分析结果
具体实现根据 FastAPI 当前最佳实践选择。

十四、数据库设计
不要设计复杂数据库。

只需要：

text
users
products
orders
如果确实有必要，可以增加：

text
categories
但不要为了体现功能增加十几张表。

数据准备：

text
几千条模拟订单数据
即可。

需要提供：

text
schema.sql
data.sql
保证项目可以快速启动。

十五、RAG知识库设计
建议：

text
Chroma
或者：

text
FAISS
优先选择更容易理解和部署的方案。

知识来源可以直接维护：

text
data/knowledge/
例如：

text
schema/
    users.md
    products.md
    orders.md

business/
    sales.md
    refund.md
启动时：

text
读取知识
↓
Embedding
↓
写入Vector Store
不要依赖复杂外部向量数据库。

十六、LLM设计
使用统一 LLM 接口。

优先：

text
Qwen
或者：

text
DeepSeek
使用 OpenAI-compatible API。

通过：

text
.env
配置：

text
LLM_API_KEY
LLM_BASE_URL
LLM_MODEL
不能把 API Key 写死在代码中。

十七、项目目录
最终目录建议：

text
ai-data-agent/
│
├── app/
│   ├── main.py
│
│   ├── api/
│   │   └── chat.py
│
│   ├── agent/
│   │   ├── graph.py
│   │   ├── state.py
│   │   └── nodes/
│   │       ├── intent.py
│   │       ├── schema_recall.py
│   │       ├── sql_generate.py
│   │       ├── sql_execute.py
│   │       └── report.py
│
│   ├── tools/
│   │   ├── schema_tool.py
│   │   └── sql_tool.py
│
│   ├── rag/
│   │   ├── documents.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── retriever.py
│
│   ├── llm/
│   │   └── client.py
│
│   ├── database/
│   │   ├── connection.py
│   │   └── repository.py
│
│   ├── security/
│   │   └── sql_validator.py
│
│   └── config.py
│
├── data/
│   ├── knowledge/
│   │   ├── schema/
│   │   └── business/
│   └── sql/
│       ├── schema.sql
│       └── data.sql
│
├── frontend/
│
├── tests/
│
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
如果现有项目结构能够更合理地复用，则可以适当调整。

不要为了完全照搬这个目录而强行重构。

十八、前端
前端只做一个简单聊天页面。

必须支持：

text
输入问题
↓
发送
↓
SSE实时显示过程
↓
显示最终答案
可以使用：

text
Vue 3
不要做复杂后台管理系统。

不要花大量时间在 CSS。

十九、异常处理
至少处理：

text
LLM调用失败
RAG没有找到Schema
SQL生成失败
SQL安全校验失败
MySQL执行失败
SSE连接异常
SQL生成失败可以提供：

text
最多2~3次重试
不要设置10次以上重试。

二十、日志
使用 Python 标准 logging。

记录：

text
用户问题
Agent节点
RAG检索结果
生成SQL
SQL校验结果
执行耗时
最终结果
异常信息
但是：

不要记录 API Key、密码等敏感信息。

二十一、测试
不做复杂 Evaluation 平台。

但是必须有最基本的测试。

至少测试：

SQL安全
text
SELECT → 通过

DELETE → 拒绝

UPDATE → 拒绝

DROP → 拒绝
RAG
text
销售额
→ orders / products
Text-to-SQL
至少准备：

text
查询销售额
查询销量最高商品
查询平均订单金额
查询某城市用户数量
查询月度销售额
多轮对话
text
查询2026年销售额
↓
那2025年呢？
二十二、README
最终 README 不要写成纯技术堆砌。

重点说明：

项目定位
text
轻量级AI数据分析Agent
核心架构
text
LLM
+
RAG
+
LangGraph
+
Tool Calling
+
Text-to-SQL
核心流程
text
用户
 ↓
Intent
 ↓
Schema RAG
 ↓
SQL Agent
 ↓
SQL Tool
 ↓
MySQL
 ↓
Result Analysis
 ↓
SSE
技术难点
重点描述：

text
1. 如何利用RAG降低Schema上下文长度
2. 如何让LLM生成可靠SQL
3. 如何控制Agent Tool权限
4. 如何保证SQL执行安全
5. 如何实现Agent过程的流式输出
二十三、面试导向
整个项目必须能够让我回答以下问题。

LLM
text
1. 项目为什么使用大模型？
2. Prompt怎么设计？
3. 如何约束LLM输出？
RAG
text
4. 为什么需要RAG？
5. RAG存的是什么？
6. Embedding是什么？
7. 相似度检索怎么工作的？
8. 为什么不把所有Schema直接放Prompt？
Agent
text
9. 什么是Agent？
10. LangGraph解决什么问题？
11. 为什么使用LangGraph？
12. State是什么？
13. Node是什么？
14. Tool Calling怎么实现？
Text-to-SQL
text
15. 自然语言怎么变成SQL？
16. Schema怎么提供给LLM？
17. SQL生成错误怎么办？
18. SQL执行如何保证安全？
后端
text
19. 为什么使用FastAPI？
20. SSE是什么？
21. SSE和WebSocket有什么区别？
22. 多轮对话如何维护？
二十四、代码要求
代码必须：

text
简单
清晰
模块化
有注释
容易调试
不要为了体现高级而：

text
过度抽象
设计大量接口
设计复杂继承体系
引入大量第三方库
优先：

能运行 + 能理解 + 能解释。

二十五、开发方式
非常重要：

不要一次性生成整个项目。

必须分阶段开发。

按照下面顺序：

text
Phase 1
项目结构 + FastAPI + LLM
        ↓
Phase 2
MySQL + 数据
        ↓
Phase 3
Text-to-SQL
        ↓
Phase 4
SQL安全
        ↓
Phase 5
RAG
        ↓
Phase 6
LangGraph
        ↓
Phase 7
Tool Calling
        ↓
Phase 8
多轮对话
        ↓
Phase 9
SSE
        ↓
Phase 10
简单前端
        ↓
Phase 11
测试
        ↓
Phase 12
Docker + README
每完成一个 Phase：

先运行
验证
解释核心代码
再进入下一阶段
二十六、每次修改代码之前
必须先告诉我：

text
1. 当前项目情况
2. 本阶段目标
3. 要修改哪些文件
4. 为什么修改
5. 技术选择原因
6. 修改后的运行方式
7. 如何验证
然后再修改代码。

不要无解释地大规模修改项目。

二十七、禁止事项
禁止：

text
1. 不分析现有代码就删除项目
2. 一次性重写整个项目
3. 添加我没有要求的复杂技术
4. 为了简历虚构功能
5. 虚构性能指标
6. 虚构测试结果
7. 把不存在的功能写进README
8. 把API Key写入代码
9. 把数据库密码写入代码
10. 使用过于复杂的Agent架构
如果现有项目某个功能无法直接迁移：

先说明原因，再给出最简单的 Python 替代方案。

二十八、最终验收标准
项目完成后，必须能够完成下面完整流程：

text
用户：

查询2026年销售额最高的商品

        ↓

FastAPI

        ↓

LangGraph

        ↓

IntentNode

        ↓

Schema RAG

        ↓

找到 orders + products + 销售额业务规则

        ↓

SQLGenerateNode

        ↓

生成SELECT SQL

        ↓

SQL Validator

        ↓

通过

        ↓

SQL Execute Tool

        ↓

MySQL

        ↓

查询结果

        ↓

ReportNode

        ↓

LLM分析

        ↓

SSE

        ↓

用户得到自然语言结果
并且支持：

text
查询2026年的销售额
        ↓
那2025年呢？
能够正确理解上下文。

二十九、最终求职定位
项目完成后，不要把它包装成：

“一个简单的ChatGPT聊天项目”

而应该定位为：

面向业务数据库的 AI Agent 数据分析系统

核心能力：

text
Python后端
+
LLM应用
+
RAG
+
Agent Workflow
+
Tool Calling
+
Text-to-SQL
+
数据库
+
SSE
最终目标：

让这个项目能够成为我简历中专门体现 AI Agent 应用开发能力的项目，并与 Java 高并发项目形成技术栈互补。

三十、现在开始
现在不要直接开始大规模编码。

第一步只做：

Step 1：分析当前 Java DataAgent 项目
请先扫描当前项目，输出：

text
A. 当前项目整体架构
B. 当前核心业务流程
C. 当前StateGraph实现
D. 当前RAG实现
E. 当前Text-to-SQL实现
F. 当前SQL执行和安全机制
G. 当前SSE实现
H. 当前数据库结构
I. 当前前端结构
J. Java → Python迁移映射表
K. 哪些代码值得复用设计
L. 哪些代码应该删除
M. 哪些功能需要新增
N. 最终Python项目架构
O. Phase 1具体实施计划