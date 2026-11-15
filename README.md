# 掌柜智库 · 企业知识库 Agent（参考实现）

面向连锁零售企业的知识库 Agent：把制度、SOP、商品资料和经营数据变成可追问、可引用、带权限的多轮问答。

> 这是用于学习/面试展示的工程参考实现，不承诺开箱即用。Milvus、Elasticsearch、PostgreSQL、Redis、OCR 和模型 API 都需要真实环境或替换成 mock。

## 1. 核心能力

- 文档接入：PDF / Word / Excel / 扫描件 OCR
- 结构感知分块：保留文档、章节、页码、表格表头、权限标签
- 混合检索：BM25 + 向量召回 + RRF 融合 + Rerank 精排
- Agent 工作流：基于 LangGraph 的意图识别、检索、工具调用、结果校验、回答生成、引用校验
- 权限控制：按租户/部门/角色/文档 ACL 在检索前过滤
- 引用溯源：回答带 [1][2] 引用，后处理校验引用是否来自本次上下文
- 评测：Top-3 / Recall@K / MRR / 引用准确率 / 拒答正确率
- 可观测：结构化日志、Prometheus 指标、按节点 trace

## 2. 架构

~~~text
【离线索引】
上传/同步 → 解析(OCR/表格) → 清洗 → 结构感知分块 → 元数据
        → Embedding → Milvus / Elasticsearch 索引

【在线问答】
用户问题 → 身份/权限 → Query 改写 → 混合召回(BM25 + 向量)
        → RRF → Rerank → 上下文组装
        → LangGraph：意图识别 → 检索 → 工具调用 → 校验
        → 回答生成 → 引用校验 → 回答 + 引用
        → 日志 / 评测 / 反馈
~~~

## 3. 目录

~~~text
app/
  main.py                 FastAPI 入口
  api/                    路由、请求/响应模型、依赖
  agent/                  LangGraph 状态、图、节点、提示词、工具
  rag/                    解析、分块、embedding、BM25、向量库、混合检索、rerank、引用
  services/               知识入库、权限、评测
  db/                     SQLAlchemy 模型与会话
  core/                   配置、日志
  evaluation/             指标与黄金评测集
tests/                    分块、RRF、引用校验
scripts/                  演示数据脚本
~~~

## 4. 关键接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | /api/v1/chat | 单轮/多轮问答 |
| POST | /api/v1/chat/stream | SSE 流式问答 |
| POST | /api/v1/documents | 上传文档并异步索引 |
| GET | /api/v1/documents/{doc_id} | 文档与索引状态 |
| DELETE | /api/v1/documents/{doc_id} | 删除文档与索引 |
| POST | /api/v1/admin/eval | 运行离线评测 |
| GET | /healthz | 健康检查 |

## 5. 关键设计

1. **解析优先**：PDF 双栏、页眉页脚、跨页表格、扫描件 OCR 都会直接影响召回上限。
2. **结构感知分块**：先按标题层级切，再按语义窗口切；表格转 Markdown，chunk 带表头。
3. **混合检索**：BM25 负责商品编码/制度编号等精确词，向量负责自然语言语义。
4. **RRF**：只用排名融合，不直接加 BM25 分数和余弦相似度。
5. **Rerank**：只对 TopN 候选做 Cross-encoder 精排，控制延迟。
6. **权限前置**：检索前按 ACL 过滤，绝不能在生成后删除。
7. **引用校验**：chunk 带 doc/page/section，回答后校验引用编号是否来自本次上下文。
8. **LLM 不碰数字**：商品/库存/报表工具由后端执行，模型只决定调用和解释。

## 6. 本地依赖

~~~bash
pip install -r requirements.txt
cp .env.example .env
# 启动 postgres / redis / elasticsearch / milvus
uvicorn app.main:app --reload
~~~

## 7. 说明

- 这是参考实现，外部服务调用通过适配器隔离，方便替换成公司内部服务。
- 生产使用前需要补：真实 OCR、鉴权/JWT 密钥、限流、审计、密钥管理、数据脱敏、模型评测。
- 代码中的 ACL、RRF、引用校验和分块逻辑是面试最值得展开的部分，详见对应文件注释。

## 8. License

MIT
