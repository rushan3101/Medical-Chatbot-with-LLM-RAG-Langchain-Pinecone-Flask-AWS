from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel

from sentence_transformers import CrossEncoder

from langchain_pinecone import PineconeVectorStore

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from src.helper import download_hugging_face_embeddings
from src.prompt import get_metadata_prompt,get_system_prompt


load_dotenv()

# Schema
class QueryMetadata(BaseModel):
    term: Optional[str]
    section: List[str]


# Models
meta_llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

final_llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    base_url="https://openrouter.ai/api/v1",
    streaming=True
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# Embeddings
embedding = download_hugging_face_embeddings()

# Vector Store
vector_store = PineconeVectorStore.from_existing_index(
    index_name= 'medical-chatbot',
    embedding= embedding
)   

# Metadata Extraction
metadata_prompt = ChatPromptTemplate.from_template(
    get_metadata_prompt() + "\n\nQuery: {query}"
)

metadata_chain = metadata_prompt | meta_llm.with_structured_output(QueryMetadata)



# Helpers
def deduplicate(docs):
    return list({d.page_content: d for d in docs}.values())


def rerank(query, docs, top_k=8):
    if not docs:
        return []

    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, docs), reverse=True)
    return [doc for _, doc in ranked[:top_k]]


def format_context(docs):
    return "\n\n---\n\n".join(
        [f'Page {d.metadata["page"]}: {d.page_content}' for d in docs]
    )



# RAG PIPELINE
def run_rag(query,chat_history):

    # 1. Metadata
    meta = metadata_chain.invoke({"query": query})

    # 2. Retrieval
    docs = []

    # Semantic
    docs += vector_store.as_retriever(
        search_kwargs={"k": 8}
    ).invoke(query)

    # Section filtered
    if meta.section:
        docs += vector_store.as_retriever(
            search_kwargs={
                "k": 8,
                "filter": {"section": {"$in": meta.section}}
            }
        ).invoke(query)

    # Term + Section
    if meta.term and meta.section:
        docs += vector_store.as_retriever(
            search_kwargs={
                "k": 8,
                "filter": {
                    "term": meta.term,
                    "section": {"$in": meta.section}
                }
            }
        ).invoke(query)

    # 3. Deduplicate
    docs = deduplicate(docs)

    # 4. Rerank
    docs = rerank(query, docs)

    # 5. Context
    context = format_context(docs)

    # 6. Final LLM (STREAM)
    prompt = ChatPromptTemplate.from_messages([
        ("system", get_system_prompt() + "\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}")
    ])

    chain = prompt | final_llm | StrOutputParser()

    return chain.stream({
        "query": query,
        "chat_history": chat_history,
        "context": context
    })