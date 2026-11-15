from typing import List
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from config import config

class ExpandedQuery(BaseModel):
    dense_query: str = Field(description="Expanded semantic query enriched with legal concepts")
    sparse_keywords: List[str] = Field(description="Key statutory terms, clause types, and boolean keywords")

class LegalQueryOptimizer:
    def __init__(self):
        # Uses local Ollama model
        self.llm = ChatOllama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.0
        )
        self.prompt = PromptTemplate(
            template="""You are a senior compliance counsel.
Analyze the user's compliance question and optimize it for a hybrid search system.

1. Resolve vague pronouns.
2. Formulate a dense search query with standard legal terminology (liability, breach, indemnity, termination).
3. Extract 3-5 sparse keywords (clause headings, statutory terms).

User Query: {query}

Return ONLY valid JSON matching the schema:
{{"dense_query": "...", "sparse_keywords": ["keyword1", "keyword2"]}}
""",
            input_variables=["query"]
        )
        # Ollama supports structured JSON enforcement
        self.chain = self.prompt | self.llm.with_structured_output(ExpandedQuery)

    def optimize(self, query: str) -> ExpandedQuery:
        try:
            return self.chain.invoke({"query": query})
        except Exception:
            # Fallback if local model produces slight JSON formatting variation
            return ExpandedQuery(
                dense_query=query,
                sparse_keywords=query.split()
            )