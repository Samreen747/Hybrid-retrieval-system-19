from typing import List, Dict
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

class HybridRRFRetriever:
    def __init__(self, vector_store, bm25_retriever, docstore: Dict[str, Document]):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.docstore = docstore
        # Initialize Cross-Encoder for precision reranking
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def retrieve(self, dense_query: str, sparse_keywords: List[str], top_k: int = 3) -> List[Document]:
        # 1. Dense retrieval
        dense_results = self.vector_store.similarity_search(dense_query, k=15)

        # 2. Sparse retrieval
        keyword_search_str = " ".join(sparse_keywords)
        sparse_results = self.bm25_retriever.invoke(keyword_search_str)

        # 3. Reciprocal Rank Fusion (RRF) at the Child Level
        rrf_scores = {}
        k_const = 60

        for rank, doc in enumerate(dense_results):
            cid = doc.metadata["child_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + (1.0 / (k_const + rank + 1))

        for rank, doc in enumerate(sparse_results):
            cid = doc.metadata["child_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + (1.0 / (k_const + rank + 1))

        # Sort children by RRF
        sorted_child_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # 4. Resolve to Parent Documents (De-duplicate)
        seen_parents = set()
        candidate_parents: List[Document] = []

        all_children = {d.metadata["child_id"]: d for d in dense_results + sparse_results}
        for cid in sorted_child_ids:
            parent_id = all_children[cid].metadata["parent_id"]
            if parent_id not in seen_parents and parent_id in self.docstore:
                seen_parents.add(parent_id)
                candidate_parents.append(self.docstore[parent_id])
            if len(candidate_parents) >= 8:
                break

        if not candidate_parents:
            return []

        # 5. Cross-Encoder Re-Ranking on Full Legal Parent Clauses
        pairs = [[dense_query, p.page_content] for p in candidate_parents]
        scores = self.reranker.predict(pairs)

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        final_docs = [candidate_parents[i] for i in ranked_indices[:top_k]]

        # Inject confidence score into metadata
        for i, idx in enumerate(ranked_indices[:top_k]):
            final_docs[i].metadata["relevance_score"] = float(scores[idx])

        return final_docs