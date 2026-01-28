# OpenAI RAG Course

## 背景

这是一个面向编程初学者和 AI 本科生的 RAG（检索增强生成）入门课程。课程从最基础的 HTTP 请求开始，逐步深入到 OpenAI API、向量数据库、LangChain 框架等核心技术，帮助学习者建立完整的 RAG 系统开发能力。

课程特点：
- 从零开始，无需复杂环境配置
- 面向过程编程风格，降低学习门槛
- 代码简洁，注释详细
- 逐步递进，循序渐进

## 目录结构

```
openai-rag-course/
├── course/                              # 课程主目录
│   ├── 01-requests-openai/              # Python 内置库 HTTP 请求
│   ├── 02-openai-sdk/                   # OpenAI 官方 SDK
│   ├── 03-openai-embedding/             # 文本向量化
│   ├── 04-text-embedding-similarity/    # 语义检索
│   ├── 05-chromadb-memory/              # 向量数据库
│   ├── 06-chromadb-openai/              # RAG 系统集成
│   ├── 07-langchain-integration/        # LangChain 框架
│   └── extra-learning-materials/        # 额外学习资料
│       ├── git-github-basics.md
│       ├── reading-github-repository.md
│       └── uv-package-manager.md
├── LICENSE
└── README.md
```

## 课程内容

1. **Python 内置库请求 OpenAI API** - 使用python原生方法，理解 API 调用的底层机制
2. **OpenAI SDK 使用** - 使用官方 SDK 简化开发，理解 prompt engineering 的作用
3. **文本向量化** - 学习 Embedding 技术和余弦相似度计算
4. **语义检索实践** - 基于长文本（如小说）构建语义搜索系统
5. **向量数据库** - 使用 ChromaDB 进行向量存储和检索
6. **RAG 系统集成** - 将 ChromaDB 与 OpenAI 结合构建完整的 RAG 系统
7. **LangChain 框架** - 使用 LangChain 简化 RAG 开发流程

## 开始学习

### 打开项目

1. 启动 VS Code
2. 点击 `文件` -> `打开文件夹`
3. 选择 `openai-rag-course` 目录
4. 在左侧资源管理器中浏览项目结构

### 学习路径

按照目录顺序，从 `course/01-requests-openai` 开始，每个目录下都有完整的代码和注释。建议：
- 先阅读代码和注释，理解实现逻辑
- 运行代码，观察输出结果
- 尝试修改参数，观察变化
- 完成一个章节后再进入下一个

## 额外资料

`extra-learning-materials/` 目录包含了 Git、GitHub、uv 等工具的学习资料，建议在学习课程的同时了解这些开发工具。

## License

MIT License
