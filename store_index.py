from dotenv import load_dotenv
from src.helper import load_pdf, create_documents, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

def save_sets(term_set: set, section_set: set):
    """
    Saves the given sets of terms and sections to text files.

    The function takes two sets of strings as input, one for terms and one for sections.
    It writes each set of strings to a separate text file, with one string per line.

    :param term_set: A set of strings containing terms extracted from PDF files.
    :param section_set: A set of strings containing sections extracted from PDF files.
    """
    with open("src\\term_set.txt", "w") as term_file:
        for term in term_set:
            term_file.write(f"{term}\n")

    with open("src\\section_set.txt", "w") as section_file:
        for section in section_set:
            section_file.write(f"{section}\n")

load_dotenv()

if __name__ == "__main__":

    extracted_data = load_pdf('data')
    print("Data Extracted Successfully" if extracted_data else "Data Extraction Failed")

    term_set = {item["term"] for item in extracted_data if item["term"]}
    section_set = {item["section"] for item in extracted_data if item["section"]}
    save_sets(term_set, section_set)

    documents = create_documents(extracted_data)
    print("Documents Created Successfully" if documents else "Document Creation Failed")

    embeddings = download_hugging_face_embeddings()

    pc = Pinecone()
    index_name = "medical-chatbot"

    if not pc.has_index(index_name):
        pc.create_index(
            name = index_name,
            dimension=384,  # Dimension of the embeddings
            metric= "cosine",  # Cosine similarity
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    print("Pinecone Index Created Successfully" if pc.has_index(index_name) else "Pinecone Index Creation Failed")

    vector_store = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name=index_name
)
    
    print("Vector Store Created Successfully" if vector_store else "Vector Store Creation Failed")  