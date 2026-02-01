import logging
import os
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings

from config import VECTOR_STORE_DIR


logger = logging.getLogger(__name__)


def get_chroma_client() -> chromadb.Client:
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    client = chromadb.Client(
        Settings(is_persistent=True, persist_directory=VECTOR_STORE_DIR)
    )
    return client


def get_collection(name: str = "documents"):
    client = get_chroma_client()
    return client.get_or_create_collection(name)


def reset_collection(name: str = "documents") -> None:
    client = get_chroma_client()
    try:
        client.delete_collection(name)
    except Exception:  # noqa: BLE001
        logger.info("Collection %s did not exist for reset", name)
    client.get_or_create_collection(name)


def add_documents_to_store(
    chunks: List[Dict[str, Any]],
    collection_name: str = "documents",
) -> None:
    collection = get_collection(collection_name)
    ids = []
    docs = []
    metadatas = []

    for idx, ch in enumerate(chunks):
        ids.append(f"chunk_{idx}_{ch['metadata'].get('filename','')}_{ch['metadata'].get('page_number',0)}_{ch['metadata'].get('char_start',0)}")
        docs.append(ch["text"])
        metadatas.append(ch["metadata"])

    from embeddings import embed_texts

    vectors = embed_texts(docs)

    collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=vectors)


def query_store(
    query: str,
    n_results: int = 20,
    collection_name: str = "documents",
) -> Dict[str, Any]:
    collection = get_collection(collection_name)
    from embeddings import embed_texts

    q_emb = embed_texts([query])[0]
    results = collection.query(query_embeddings=[q_emb], n_results=n_results)
    return results
