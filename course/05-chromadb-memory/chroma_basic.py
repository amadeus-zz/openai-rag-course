"""ChromaDB 内存模式基础示例(CRUD)

本节目的:
    在第4节学习了“向量 + 相似度检索”后，本节引入向量数据库 ChromaDB。
    通过“内存模式”快速体验 ChromaDB 的核心能力:建库、建 collection、
    以及对文档(含向量与元数据)的增删改查。

核心流程:
    1. 创建内存数据库(Client)
    2. 创建 collection
    3. 添加 mock_document(Create)
    4. 查询数据(Read:全量/相似度/元数据过滤)
    5. 更新数据(Update)
    6. 删除数据(Delete)

CRUD 是什么:
    - Create:添加数据
    - Read:查询数据
    - Update:更新数据
    - Delete:删除数据
    这四类操作构成了数据库最基础、最常见的使用方式。

核心概念(详细解释):
    - Client(数据库客户端):
        ChromaDB 的入口对象。你通过 Client 创建或获取 collection,
        并决定数据存放方式(内存/持久化)。
    - Collection(集合/表):
        存放“同一类数据”的容器，类似关系型数据库里的表。
        例如:一个 collection 用于存放“城市介绍”，另一个用于“产品说明”。
    - Document(文档):
        原始文本内容，是人类可读的部分。
    - Embedding(向量):
        文本的数字表示。ChromaDB 用向量做相似度检索。
    - Metadata(元数据):
        结构化字段，用于过滤与解释，比如 city、topic、author 等。
    - ID(唯一标识):
        每条数据的主键。增删改查都依赖 ID 来定位目标数据。

依赖安装:
    pip install chromadb

最小调用方式:
    python chroma_basic.py
"""

import chromadb


# -----------------------------------------------------------------------------
# 手动构造向量，便于理解
# -----------------------------------------------------------------------------
mock_documents = [
    {
        "id": "doc-001",
        "text": "天津，这座海滨直辖市在中国北方，以中西文化交汇的魅力、独特的历史景观及美食文化著称。",
        "embedding": [0.10, 0.20, 0.30, 0.40],
        "metadata": {"city": "天津", "topic": "城市", "source": "文本1"},
    },
    {
        "id": "doc-002",
        "text": "天津是一座融合中西文化的中国北方沿海直辖市，以独特的历史风貌、滨海开放气质和美食文化闻名。",
        "embedding": [0.12, 0.18, 0.33, 0.38],
        "metadata": {"city": "天津", "topic": "城市", "source": "文本2"},
    },
    {
        "id": "doc-003",
        "text": "从京城胡同里的老字号到天南海北的特色菜馆，隋坡用镜头记录美食",
        "embedding": [0.80, 0.10, 0.05, 0.05],
        "metadata": {"city": "北京", "topic": "美食", "source": "文本3"},
    },
]


def main():
    # 内存模式:程序退出后数据消失
    client = chromadb.Client()

    # 创建 collection(类似表)，用于存放同一类向量数据
    # 可通过 metadata 指定向量距离度量方式:如 cosine 或 ip
    collection = client.create_collection(
        name="rag_course_demo",
        metadata={"hnsw:space": "cosine"},
    )

    # Create:添加 mock_document(文档 + 向量 + 元数据)
    # ChromaDB 的 add 接口需要分别提供 ids/documents/embeddings/metadatas 四个列表
    ids = []
    documents = []
    embeddings = []
    metadatas = []
    for d in mock_documents:
        ids.append(d["id"])
        documents.append(d["text"])
        embeddings.append(d["embedding"])
        metadatas.append(d["metadata"])
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"[Create] 已添加 {len(mock_documents)} 条文档。")

    # Read-1:全量获取
    all_data = collection.get()
    print("[Read] 全量获取 ids:", all_data["ids"])

    # Read-2:向量相似度检索(输入一个查询向量)
    # 这一步与第4节“手写相似度检索”的核心思想一致:都在比对“查询向量 vs 文档向量”的相似度。
    # 区别在于:第4节用 numpy 手动算相似度；这里由 ChromaDB 内部完成并返回 Top-K 结果。
    query_embedding = [0.11, 0.19, 0.32, 0.39]
    query_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=2,
    )
    print("[Read] 相似度 top2 ids:", query_result["ids"][0])

    # Read-3:元数据过滤(where)
    filtered = collection.get(where={"topic": "美食"})
    print("[Read] 过滤 topic=美食 ids:", filtered["ids"])

    # Update:更新 doc-003 的文本与向量
    collection.update(
        ids=["doc-003"],
        documents=["Embedding 可以用来做语义检索与推荐。"],
        embeddings=[[0.75, 0.15, 0.05, 0.05]],
        metadatas=[{"city": "none", "topic": "向量", "source": "更新"}],
    )
    updated = collection.get(ids=["doc-003"])
    print("[Update] doc-003 document:", updated["documents"][0])

    # Delete:删除 doc-002
    collection.delete(ids=["doc-002"])
    after_delete = collection.get()
    print("[Delete] 删除 doc-002 后 ids:", after_delete["ids"])


if __name__ == "__main__":
    main()

"""
===============================================================================
思考题
===============================================================================

1. 创建 collection 的 metadata 参数中 {"hnsw:space": "cosine"} 是什么意思?为什么要这样设置?
"""
