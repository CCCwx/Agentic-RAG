"""
Multi-query retrieval and deduplication for Stage 2.
Use each query in Query_List to retrieve Top-2 from vector store, then dedupe and merge.
"""
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from utils.logger_handler import logger


def _doc_key(doc: Document) -> str:
    """Key for deduplication: content + source if present."""
    meta = doc.metadata or {}
    source = meta.get("source", "")
    return (doc.page_content or "").strip() + "|" + str(source)


def multi_query_retrieve(
    vector_store: VectorStore,
    query_list: list[str],
    k_per_query: int = 2,
) -> list[Document]:
    """
    For each query in query_list, retrieve top k_per_query docs; dedupe and merge.
    Result size is in [k_per_query, min(len(query_list) * k_per_query, unique docs)].
    PDF: 文档的数字最终范围在 [2, 4] for 2 queries with k=2.
    """
    seen_keys = set()
    merged: list[Document] = []
    retriever = vector_store.as_retriever(search_kwargs={"k": k_per_query})

    for q in query_list:
        if not (q or str(q).strip()):
            continue
        try:
            docs = retriever.invoke(q.strip())
        except Exception as e:
            logger.warning(f"[multi_query_retrieve] query '{q[:50]}...' failed: {e}")
            continue
        for doc in docs:
            key = _doc_key(doc)
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append(doc)
    return merged
