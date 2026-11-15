import os
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class LegalDocumentProcessor:
    def __init__(self, parent_chunk_size: int = 2000, child_chunk_size: int = 400):
        # Parent splitter splits along legal clause headers and sections
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=200,
            separators=["\n\nARTICLE ", "\n\nSECTION ", "\n\nClause ", "\n\n", "\n", " "]
        )
        # Child splitter generates dense vectors for precise retrieval
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=50
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = os.path.basename(file_path)
        return docs

    def process_documents(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        Returns:
            parent_docs: Full clause contexts with unique parent_id
            child_docs: Smaller chunks tagged with their parent_id
        """
        parent_docs = self.parent_splitter.split_documents(docs)
        child_docs = []

        for p_idx, p_doc in enumerate(parent_docs):
            p_id = f"{p_doc.metadata.get('source_file', 'doc')}_p_{p_idx}"
            p_doc.metadata["parent_id"] = p_id

            # Generate child documents from this parent
            sub_chunks = self.child_splitter.split_text(p_doc.page_content)
            for c_idx, sub_text in enumerate(sub_chunks):
                child_doc = Document(
                    page_content=sub_text,
                    metadata={
                        **p_doc.metadata,
                        "parent_id": p_id,
                        "child_id": f"{p_id}_c_{c_idx}"
                    }
                )
                child_docs.append(child_doc)

        return parent_docs, child_docs