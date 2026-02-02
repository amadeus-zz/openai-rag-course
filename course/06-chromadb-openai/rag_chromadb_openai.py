"""第6节:把 ChromaDB 与 OpenAI 组合成一个最小可用的 RAG 系统

本节目标
-----------------------------------------------------------------------------
把前面几节已经出现过的"零件"组装成一条完整链路:

1) 示例长文本:魔戒节选(course/04-text-embedding-similarity/data/魔戒节选.txt)
2) splitter:把长文本切成多个 chunk(固定长度 + overlap)
3) OpenAI embedding:把每个 chunk 变成向量
4) Persistent Chroma Client:把 (chunk_text, embedding, metadata) 持久化存起来
5) top3 召回:用户提问 -> 向量化 -> 在 ChromaDB 里检索 top3 chunk
6) 生成答案:把 top3 chunk 作为上下文,交给 OpenAI ChatCompletion 生成回答

你会得到一个"能跑通,能复现,能重复运行"的最小 RAG Demo.

依赖安装
-----------------------------------------------------------------------------
pip install openai chromadb python-dotenv

环境变量(放在项目根目录的 .env)
-----------------------------------------------------------------------------

最小运行
-----------------------------------------------------------------------------
python course/06-chromadb-openai/rag_chromadb_openai.py
"""

import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


# -----------------------------------------------------------------------------
# 配置与常量
# -----------------------------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

TEXT_FILE_PATH = Path("course/04-text-embedding-similarity/data/魔戒节选.txt")
CHROMA_PERSIST_DIR = Path("course/06-chromadb-openai/chroma_db")
COLLECTION_NAME = "lotr_rag_demo"


class Chunk:
    def __init__(self, chunk_id: str, text: str, start: int, end: int):
        self.chunk_id = chunk_id
        self.text = text
        self.start = start
        self.end = end


def _require_env(name: str, value: Optional[str]) -> str:
    if value is None or value.strip() == "":
        raise RuntimeError(
            f"缺少环境变量 {name}.请在 .env 中设置 {name}=... 后再运行."
        )
    return value


def split_text_fixed(full_text: str, chunk_size: int, overlap: int) -> List[Chunk]:
    """固定长度切分:简单,可控,方便教学.

    - chunk_size: 每个 chunk 的字符数
    - overlap: 相邻 chunk 的重叠字符数,避免切断语义导致召回不稳定
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须 > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须满足 0 <= overlap < chunk_size")

    chunks: List[Chunk] = []
    start = 0
    i = 0
    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunk_text = full_text[start:end]
        chunk_id = f"lotr-chunk-{i:05d}"
        chunks.append(Chunk(chunk_id=chunk_id, text=chunk_text, start=start, end=end))
        i += 1
        start += chunk_size - overlap
    return chunks


def batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size 必须 > 0")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def embed_texts_openai(client: OpenAI, texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """用 OpenAI Embeddings API 把一组文本变成向量.

    说明:
    - API 支持一次传入多个 input,这样比每段单独调用更省时,也更稳定.
    """
    embeddings: List[List[float]] = []
    for batch in batched(texts, batch_size=batch_size):
        response = client.embeddings.create(model=EMBEDDING_MODEL_NAME, input=batch)
        # response.data 的顺序与 input 一致
        for item in response.data:
            embeddings.append(item.embedding)
    return embeddings


def build_or_load_collection(persist_dir: Path):
    """使用持久化模式的 ChromaDB Client.

    - persist_dir: 存储目录.
    """
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


class RAGPipeline:
    """把本节的几个步骤(索引,召回,生成)封装成一个对象."""

    def __init__(self, openai_client: OpenAI, collection):
        self.openai_client = openai_client
        self.collection = collection

    def ensure_index(self, chunks: List[Chunk], source_name: str):
        index_chunks_if_needed(
            collection=self.collection,
            openai_client=self.openai_client,
            chunks=chunks,
            source_name=source_name,
        )

    def retrieve(self, question: str, top_k: int) -> List[Tuple[str, str, float]]:
        return retrieve_top_k(
            collection=self.collection,
            openai_client=self.openai_client,
            query=question,
            top_k=top_k,
        )

    def generate(self, question: str, retrieved: List[Tuple[str, str, float]]) -> str:
        return answer_with_context(
            openai_client=self.openai_client,
            question=question,
            retrieved=retrieved,
        )


def index_chunks_if_needed(collection, openai_client: OpenAI, chunks: List[Chunk], source_name: str):
    """把文本 chunks 写入 ChromaDB(如果该 collection 还是空的)."""
    existing_count = collection.count()
    if existing_count > 0:
        print("[Index] 检测到 collection 已有数据(count=" + str(existing_count) + "),跳过重新写入.")
        return

    print("[Index] collection 为空,开始写入 " + str(len(chunks)) + " 个文本块 ...")

    text_list: List[str] = []
    for chunk in chunks:
        text_list.append(chunk.text)

    embedding_list = embed_texts_openai(openai_client, text_list, batch_size=32)

    id_list: List[str] = []
    metadata_list: List[dict] = []
    document_list: List[str] = []

    for chunk in chunks:
        id_list.append(chunk.chunk_id)
        document_list.append(chunk.text)
        metadata_list.append(
            {
                "source": source_name,
                "chunk_index": int(chunk.chunk_id.split("-")[-1]),
                "start_char": chunk.start,
                "end_char": chunk.end,
            }
        )

    collection.add(
        ids=id_list,
        documents=document_list,
        embeddings=embedding_list,
        metadatas=metadata_list,
    )
    print("[Index] 写入完成.")


def retrieve_top_k(collection, openai_client: OpenAI, query: str, top_k: int) -> List[Tuple[str, str, float]]:
    """向量化 query,然后在 ChromaDB 里做 top-k 检索.

    返回:[(chunk_id, chunk_text, distance), ...]
    - 在 cosine 空间里,distance 越小越相近(越相关)
    """
    query_embedding_list = embed_texts_openai(openai_client, [query], batch_size=1)
    query_embedding = query_embedding_list[0]

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"],
    )

    id_list = result["ids"][0]
    document_list = result["documents"][0]
    distance_list = result["distances"][0]

    hits: List[Tuple[str, str, float]] = []
    index = 0
    while index < len(id_list):
        chunk_id = id_list[index]
        chunk_text = document_list[index]
        distance = float(distance_list[index])
        hits.append((chunk_id, chunk_text, distance))
        index = index + 1

    return hits


def answer_with_context(openai_client: OpenAI, question: str, retrieved: List[Tuple[str, str, float]]) -> str:
    """把召回结果拼成上下文,然后让 ChatCompletion 生成答案."""
    context_lines: List[str] = []
    for index in range(len(retrieved)):
        rank = index + 1
        chunk_id = retrieved[index][0]
        chunk_text = retrieved[index][1]
        context_lines.append("[Chunk " + str(rank) + " | " + chunk_id + "]\n" + chunk_text)
    context = "\n\n---\n\n".join(context_lines)

    system_prompt = (
        "你是一个严谨的助教.你只能使用给定的 Context 回答问题."
        "如果 Context 不足以回答,直接说'材料不足',并指出缺少什么信息."
        "回答尽量简洁,但不要省略关键推理."
    )

    user_prompt = "Question:\n" + question + "\n\nContext:\n" + context

    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def main():
    _require_env("API_KEY", API_KEY)
    _require_env("MODEL_NAME", MODEL_NAME)
    _require_env("EMBEDDING_MODEL_NAME", EMBEDDING_MODEL_NAME)

    print("=" * 70)
    print("步骤1:读取示例长文本(魔戒节选)")
    print("=" * 70)

    if not TEXT_FILE_PATH.exists():
        raise FileNotFoundError(f"找不到示例文本文件:{TEXT_FILE_PATH}")

    full_text = TEXT_FILE_PATH.read_text(encoding="utf-8")
    print(f"文本路径: {TEXT_FILE_PATH}")
    print(f"文本字符数: {len(full_text)}")
    print(f"文本预览: {full_text[:120]}...")

    print("\n" + "=" * 70)
    print("步骤2:splitter 切分长文本")
    print("=" * 70)

    chunk_size = 1000
    overlap = 80
    chunks = split_text_fixed(full_text, chunk_size=chunk_size, overlap=overlap)
    print("切分参数: chunk_size=" + str(chunk_size) + ", overlap=" + str(overlap))
    print("切分结果: " + str(len(chunks)) + " 个文本块")

    preview_count = 2
    if len(chunks) < preview_count:
        preview_count = len(chunks)

    for index in range(preview_count):
        chunk = chunks[index]
        preview_text = chunk.text[:80] + "..."
        print("- chunk[" + str(index) + "] id=" + chunk.chunk_id + " chars=[" + str(chunk.start) + "," + str(chunk.end) + ") preview=" + preview_text)

    print("\n" + "=" * 70)
    print("步骤3:创建 OpenAI Client(用于 embedding + 生成回答)")
    print("=" * 70)

    # BASE_URL 允许为空:如果你用官方 OpenAI,base_url 可以不传.
    if BASE_URL and BASE_URL.strip():
        openai_client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    else:
        openai_client = OpenAI(api_key=API_KEY)
    print("OpenAI Client 已就绪.")

    print("\n" + "=" * 70)
    print("步骤4:创建 ChromaDB PersistentClient + collection")
    print("=" * 70)

    collection = build_or_load_collection(CHROMA_PERSIST_DIR)
    print("Chroma persist dir: " + str(CHROMA_PERSIST_DIR))
    print("Collection name: " + str(COLLECTION_NAME))

    print("\n" + "=" * 70)
    print("步骤5:索引写入(第一次运行会写入;后续运行会复用)")
    print("=" * 70)

    pipeline = RAGPipeline(openai_client=openai_client, collection=collection)

    pipeline.ensure_index(
        chunks=chunks,
        source_name="魔戒节选",
    )

    print("\n" + "=" * 70)
    print("步骤6:用户提问 -> top3 召回")
    print("=" * 70)

    # 例如:question = "甘道夫是谁?"
    question = "托尔金的生平是什么?"
    top_k = 3
    retrieved = pipeline.retrieve(question=question, top_k=top_k)

    print("Question: " + question)

    # distance 与余弦相似度
    #   cosine_distance(x, y) = 1.0 - dot(x, y) / (||x|| * ||y|| + eps)
    # 其中 dot(x, y) / (||x|| * ||y||) 就是常见的余弦相似度 cosine_similarity.
    # 因此:distance 越小  <=>  similarity 越大  <=> 语义越相近.

    print("Top-" + str(top_k) + " 召回结果(distance 越小越相关):")

    index = 0
    while index < len(retrieved):
        rank = index + 1
        chunk_id = retrieved[index][0]
        chunk_text = retrieved[index][1]
        distance = retrieved[index][2]

        preview = chunk_text
        if len(preview) > 160:
            preview = preview[:160] + "..."

        print("- rank=" + str(rank) + " id=" + chunk_id + " distance=" + format(distance, ".4f") + " preview=" + preview)
        index = index + 1

    print("\n" + "=" * 70)
    print("步骤7:把 top3 作为上下文,让 LLM 生成回答(RAG 的 Generation 阶段)")
    print("=" * 70)

    answer = pipeline.generate(question=question, retrieved=retrieved)
    print("\n回答:")
    print(answer)

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(
        "你已经跑通了一个最小 RAG:\n"
        "- splitter 把长文切块\n"
        "- embedding 把块变成向量\n"
        "- ChromaDB 持久化保存向量与文档\n"
        "- query 时 top3 召回\n"
        "- LLM 用召回上下文生成答案\n"
    )


if __name__ == "__main__":
    main()
