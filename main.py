from dotenv import load_dotenv
from typing import List, Optional, Generator
from pydantic import BaseModel

from sentence_transformers import CrossEncoder

from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage

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
meta_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

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
    get_metadata_prompt()
)

metadata_chain = metadata_prompt | meta_llm.with_structured_output(QueryMetadata)



# Helpers
def deduplicate(docs: List[Document]) -> List[Document]:
    """
    Remove duplicates from a list of documents based on their page content.

    Args:
        docs (list[Document]): A list of Document objects.

    Returns:
        list[Document]: A list of Document objects with no duplicates based on their page_content.
    """
    return list({d.page_content: d for d in docs}.values())


def rerank(query: str, docs: List[Document], top_k: int = 8) -> List[Document]:
    """
    Re-rank a list of documents based on their similarity to a given query.

    Args:
        query (str): The query to rank the documents against.
        docs (list[Document]): A list of Document objects to rank.
        top_k (int, optional): The number of top-ranked documents to return. Defaults to 8.

    Returns:
        list[Document]: A list of the top-ranked Document objects.
    """
    if not docs:
        return []

    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, docs), reverse=True)
    return [doc for _, doc in ranked[:top_k]]


def format_context(docs: List[Document]) -> str:
    """
    Format a list of documents into a string to be used as context for a language model.

    Args:
        docs (list[Document]): A list of Document objects to format.

    Returns:
        str: A formatted string containing the page number and page content for each document.
    """
    return "\n\n---\n\n".join(
        [d.page_content for d in docs]
    )



# RAG PIPELINE
def run_rag(query: str,chat_history:list[BaseMessage]) -> str:
    """
    Run the RAG pipeline to generate a response to a given query.

    The pipeline consists of the following steps:

    1. Metadata Extraction: Extract term and section metadata from the query using a Google Generative AI model.

    2. Retrieval: Retrieve documents from the vector store based on the query, using semantic search, section filtered search, and term + section search.

    3. Deduplicate: Remove duplicates from the list of retrieved documents based on their page content.

    4. Rerank: Re-rank the list of documents based on their similarity to the query using a CrossEncoder model.

    5. Context: Format the top-ranked documents into a string to be used as context for the final language model.

    6. Final LLM (STREAM): Use the context and query to generate a response using a ChatOpenAI model.

    Args:
        query (str): The query to generate a response to.
        chat_history (list[dict]): The conversation history to use as context.

    Returns:
        generator: A generator that yields the response to the query.
    """

    # 1. Metadata
    query_chat_history = [history for history in chat_history if isinstance(history, HumanMessage)]
    meta = metadata_chain.invoke({"query": query, "query_history": query_chat_history})

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
    ranked_docs = rerank(query, docs)

    # 5. Context
    context = format_context(ranked_docs)

    # 6. Final LLM (STREAM)
    prompt = ChatPromptTemplate.from_messages([
        ("system", get_system_prompt() + "\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}")
    ])

    chain = prompt | final_llm | StrOutputParser()

    return chain.invoke({
        "query": query,
        "chat_history": chat_history,
        "context": context
    })