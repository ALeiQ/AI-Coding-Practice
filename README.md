# AI-Coding-Practice

A collection of hands-on AI coding projects, covering RAG, agents, SQL, fine-tuning, and more.

好选择。作为资深工程师，你的优势是工程能力，AI 面试最看重的是实战落地能力而非调 API。给你按"从会用到深入"排一组课题：

### 第一阶段：基础应用（各 1~3 天，做熟 TS/API 调用）

- RAG 知识库问答 —— 用 Pinecone/Qdrant 向量库 + 文档切分 + 检索增强，回答本地文档的问题。考察点：chunk 策略、embedding 选型、混合检索。
- Function Calling / Tool Use —— 做一个能调用天气、数据库查询、计算器等工具的 agent。考察点：结构化输出、工具注册、多轮工具调用。
- SQL→自然语言 —— 用户问中文，自动生成并执行 SQL 查询返回结果（带权限和护栏）。

### 第二阶段：Agent 与工作流（3~5 天）

- 多智能体协作 —— 如"产品经理 agent + 工程师 agent + 评审 agent"协作完成一个需求，用 LangGraph/CrewAI 或纯自研编排。
- ReAct Agent + 记忆系统 —— 支持长期记忆（写进向量库）、短期记忆（会话摘要）的 agent。
- 自动化测试生成器 —— 输入一个函数，自动写单元测试并运行验证。面试亮点：闭环（AI 生成→执行→反馈→修正）。

### 第三阶段：深入底层（拉开面试差距）

- 手写一个 RAG 框架（不依赖 LlamaIndex）—— 自己实现切分、检索、重排序、流式输出，能讲清楚每个环节。
- 微调实践 —— 用 LoRA 微调一个小模型（如 Qwen），针对你自己的数据集，对比微调前后效果。
- 推理加速 / 缓存 —— 语义缓存（相同问题直接返回）、KV-Cache 或流式传输优化。

### 面试加分项

- 每个项目都要有 **评估体系**：像 RAG 就写 evals（忠实度、命中率），这是现在面试官最看重的。
- 做 1 个有 **生产意识** 的项目：加日志、成本追踪、错误重试、并发控制。
- 写 **架构图 + README**，讲清楚 trade-off（为什么选这个切分方式、为什么这个向量库）。

建议顺序：1 → 3 → 4 → 7 → 8，九个课题里挑 3~4 个做深做透，比全做更出彩。

想从哪个开始？如果没偏好，我建议先做 1（RAG 知识库），它是所有 AI 应用的基石。确定后我就帮你从零搭起来。

## Projects

| # | Project | Description |
|---|---------|-------------|
| 01 | [RAG-Knowledge-QA](01-RAG-Knowledge-QA/) | RAG-based knowledge QA with Qdrant hybrid retrieval |
| 02 | [Function-Calling-Agent](02-Function-Calling-Agent/) | Agent with function/tool calling capabilities |
| 03 | [SQL-to-NL](03-SQL-to-NL/) | Natural language to SQL translation |
| 04 | [Multi-Agent-Collaboration](04-Multi-Agent-Collaboration/) | Multi-agent collaborative task solving |
| 05 | [ReAct-Agent-Memory](05-ReAct-Agent-Memory/) | ReAct agent with memory integration |
| 06 | [Auto-Test-Generator](06-Auto-Test-Generator/) | Automated test case generation |
| 07 | [Handwritten-RAG-Framework](07-Handwritten-RAG-Framework/) | From-scratch RAG framework implementation |
| 08 | [LoRA-Finetuning](08-LoRA-Finetuning/) | LoRA parameter-efficient fine-tuning |
| 09 | [Inference-Cache](09-Inference-Cache/) | LLM inference caching and optimization |

## Tech Stack

- **Python 3.9+**
- **Qdrant** — Vector database
- **Ollama** — Local LLM inference
- **FastEmbed** — Local embeddings
- **LangChain** — Text processing

## License

MIT
