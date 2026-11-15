import streamlit as st
import os
import tempfile
from ingestion.chunking import LegalDocumentProcessor
from ingestion.indexer import HybridLegalIndexer
from pipeline.query_expansion import LegalQueryOptimizer
from pipeline.hybrid_retriever import HybridRRFRetriever
from pipeline.generator import ComplianceGenerator
from config import config

st.set_page_config(page_title="Legal & Regulatory RAG", layout="wide")
st.title("⚖️ Multi-Vector Hybrid Compliance Assistant")

# Initialize persistent session states
if "indexer" not in st.session_state:
    st.session_state.indexer = HybridLegalIndexer(config.CHROMA_PERSIST_DIR, config.DOCSTORE_PATH)
    if os.path.exists(os.path.join(config.DOCSTORE_PATH, "parent_store.pkl")):
        st.session_state.indexer.load_indexes()
        st.session_state.indexed = True
    else:
        st.session_state.indexed = False

# Sidebar: Document Ingestion
with st.sidebar:
    st.header("📂 Document Ingestion")
    uploaded_files = st.file_uploader("Upload Legal PDFs/Policies", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process & Index Documents"):
        if uploaded_files:
            with st.spinner("Extracting, segmenting, and indexing parent/child vectors..."):
                processor = LegalDocumentProcessor(config.PARENT_CHUNK_SIZE, config.CHILD_CHUNK_SIZE)
                all_parents, all_children = [], []
                
                for uf in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uf.read())
                        tmp_path = tmp.name
                    
                    raw_docs = processor.load_pdf(tmp_path)
                    # Restore original uploaded filename
                    for d in raw_docs:
                        d.metadata["source_file"] = uf.name
                    
                    parents, children = processor.process_documents(raw_docs)
                    all_parents.extend(parents)
                    all_children.extend(children)
                    os.remove(tmp_path)
                
                st.session_state.indexer.build_indexes(all_parents, all_children)
                st.session_state.indexed = True
                st.success(f"Indexed {len(all_parents)} parent clauses ({len(all_children)} vector chunks).")
        else:
            st.error("Upload at least one document.")

# Main Interface
query = st.text_input("Enter compliance inquiry or contract question:", placeholder="e.g., What are our indemnity obligations in case of IP infringement?")

if st.button("Evaluate Compliance") and query:
    if not st.session_state.indexed:
        st.error("Please index documents in the sidebar first.")
    else:
        with st.status("Executing Multi-Stage Hybrid RAG...", expanded=True) as status:
            # 1. Query Rewriting
            st.write("🔍 Expanding & resolving legal terminology...")
            optimizer = LegalQueryOptimizer()
            expanded_query = optimizer.optimize(query)
            st.caption(f"**Dense Target:** {expanded_query.dense_query}")
            st.caption(f"**Sparse Keywords:** {', '.join(expanded_query.sparse_keywords)}")

            # 2. Hybrid Retrieval + Reranking
            st.write("⚡ Hybrid Retrieval (Chroma + BM25) & Cross-Encoder Reranking...")
            retriever = HybridRRFRetriever(
                vector_store=st.session_state.indexer.vector_store,
                bm25_retriever=st.session_state.indexer.bm25_retriever,
                docstore=st.session_state.indexer.docstore
            )
            top_clauses = retriever.retrieve(
                dense_query=expanded_query.dense_query,
                sparse_keywords=expanded_query.sparse_keywords,
                top_k=config.TOP_K_RERANK
            )

            # 3. Grounded Generation
            st.write("⚖️ Performing compliance synthesis & citation verification...")
            generator = ComplianceGenerator()
            response = generator.generate(query, top_clauses)
            status.update(label="Analysis Completed!", state="complete", expanded=False)

        # Output Layout
        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("📋 Legal Verdict")
            if not response.is_grounded:
                st.warning("⚠️ Insufficient context in provided documents to substantiate a risk-free verdict.")
            
            st.markdown(f"**Conclusion:** {response.compliance_verdict}")
            st.markdown("### Analysis")
            st.write(response.legal_analysis)

        with col2:
            st.subheader("📌 Cited Authorities")
            for cit in response.citations:
                with st.expander(f"Clause Source: {cit.source_file} (Page {cit.page_number})"):
                    st.markdown(f"*{cit.verbatim_quote}*")

            st.subheader("📄 Top Retrieved Context Chunks")
            for i, doc in enumerate(top_clauses):
                score = doc.metadata.get("relevance_score", 0.0)
                with st.expander(f"Context Chunk #{i+1} (Rerank Score: {score:.4f})"):
                    st.write(doc.page_content)