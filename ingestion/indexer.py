import os
import pickle
from typing import List, Dict
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from config import config

class HybridLegalIndexer:
    def __init__(self, persist_dir: str, docstore_path: str):
        self.persist_dir = persist_dir
        self.docstore_path = docstore_path
        
        # 100% Free, runs locally on CPU/GPU
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"} # change to "cuda" or "mps" if you have a GPU
        )
        self.vector_store = None
        self.bm25_retriever = None
        self.docstore: Dict[str, Document] = {}

    def build_indexes(self, parent_docs: List[Document], child_docs: List[Document]):
        os.makedirs(self.persist_dir, exist_ok=True)
        os.makedirs(self.docstore_path, exist_ok=True)

        # 1. Store Full Parent Clauses
        for doc in parent_docs:
            self.docstore[doc.metadata["parent_id"]] = doc

        with open(os.path.join(self.docstore_path, "parent_store.pkl"), "wb") as f:
            pickle.dump(self.docstore, f)

        # 2. Build Dense Index with local HuggingFace embeddings
        self.vector_store = Chroma.from_documents(
            documents=child_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )

        # 3. Build Sparse BM25 Index
        self.bm25_retriever = BM25Retriever.from_documents(child_docs)
        self.bm25_retriever.k = config.TOP_K_RETRIEVAL

        with open(os.path.join(self.docstore_path, "bm25_index.pkl"), "wb") as f:
            pickle.dump(self.bm25_retriever, f)

    def load_indexes(self):
        with open(os.path.join(self.docstore_path, "parent_store.pkl"), "rb") as f:
            self.docstore = pickle.load(f)

        self.vector_store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )

        with open(os.path.join(self.docstore_path, "bm25_index.pkl"), "rb") as f:
            self.bm25_retriever = pickle.load(f)