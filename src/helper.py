import fitz  # PyMuPDF
import re
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

def load_pdf(data_dir: str) -> list[dict]: 
    """
    Load PDF files from the given directory and extract structured data from them.
    
    Args:
        data_dir (str): The directory containing the PDF files.
    
    Returns:
        list[dict]: A list of dictionaries containing the extracted structured data.
        Example dictionary format:
        {
            "term": "Diabetes",
            "section": "Symptoms",
            "content": "Common symptoms include increased thirst, frequent urination, and fatigue.",
            "page": 5
        }
    """
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    final_structured_data = []
    
    for pdf_file in pdf_files:
        doc = fitz.open(os.path.join(data_dir, pdf_file))
        structured_data = extract_structured_data(doc)
        final_structured_data.extend(structured_data)
    
    return final_structured_data

def clean_text(text: str) -> str:
    """
    Clean the given text by replacing any whitespace characters with a single space and removing leading and trailing whitespace.
    
    Args:
        text (str): The text to be cleaned.
    
    Returns:
        str: The cleaned text.
    """
    return re.sub(r"\s+", " ", text).strip()

def extract_structured_data(doc: fitz.Document) -> list[dict]:
    """
    Extract structured data from a given fitz PDF document.

    The function takes a fitz PDF document as input and returns a list of dictionaries containing the extracted structured data.

    Args:
        doc (fitz.Document): The fitz PDF document to extract structured data from.

    Returns:
        list[dict]: A list of dictionaries containing the extracted structured data.

    Example dictionary format:
    {
        "term": "Diabetes",
        "section": "Symptoms",
        "content": "Common symptoms include increased thirst, frequent urination, and fatigue.",
        "page": 5
    }
    """
    structured_data = []

    state = {
        "term": None,
        "section": None,
        "prev_section": None,
        "in_key_terms": False,
        "section_buffer": [],
        "key_terms_buffer": []
    }


    def save_section(page_num):
        term = state["term"]
        section = state["section"]

        if not term or not section:
            return

        buffer = (
            state["key_terms_buffer"]
            if state["in_key_terms"]
            else state["section_buffer"]
        )

        content = " ".join(buffer).strip()
        if not content:
            return

        structured_data.append({
            "term": term,
            "section": section,
            "content": content,
            "page": page_num + 1
        })

        # Clear buffer
        if state["in_key_terms"]:
            state["key_terms_buffer"].clear()
        else:
            state["section_buffer"].clear()


    for page_num, page in enumerate(doc):

        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):

                    text = clean_text(span["text"])
                    if not text or text == "GALE ENCYCLOPEDIA OF MEDICINE 2" or text.isdigit():
                        continue

                    font_size = span["size"]
                    font_name = span["font"]

                    # -------- TERM --------
                    if font_size == 15:
                        save_section(page_num)
                        state["term"] = text
                        state["section"] = None
                        continue

                    # -------- KEY TERMS START --------
                    if font_size == 12.5 and text == "KEY TERMS":
                        state["prev_section"] = state["section"]
                        state["section"] = "KEY TERMS"
                        state["in_key_terms"] = True
                        continue

                    # -------- SECTION --------
                    if font_size == 11 or (text == "Resources" and font_name == "Optima-Bold"):

                        # If exiting KEY TERMS
                        if state["in_key_terms"]:
                            save_section(page_num)
                            state["in_key_terms"] = False
                            state["section"] = state["prev_section"]

                        # Save previous section if exists
                        save_section(page_num)

                        state["prev_section"] = state["section"]
                        state["section"] = text
                        continue

                    # -------- KEY TERMS END --------
                    if state["in_key_terms"] and font_name == "Times-Roman":
                        save_section(page_num)
                        state["in_key_terms"] = False
                        state["section"] = state["prev_section"]
                        continue

                    # -------- CONTENT --------
                    if state["term"] and state["section"]:
                        if state["in_key_terms"]:
                            state["key_terms_buffer"].append(text)
                        else:
                            state["section_buffer"].append(text)

    # Final flush
    save_section(page_num)

def create_documents(structured_data: list[dict]) -> list[Document]:
    """
    Create a list of langchain Document objects from a given list of structured data.

    The function takes a list of dictionaries containing structured data as input and returns a list of Document objects.

    Args:
        structured_data (list[dict]): A list of dictionaries containing structured data in the format of:
            {
                "term": str,
                "section": str,
                "content": str,
                "page": int
            }

    Returns:
        list[Document]: A list of langchain Document objects containing the structured data with term ,section and page number as metadata and content as page_content.
        Example Document format:
        Document(
            page_content="Page: 5 -> Diabetes -> Symptoms ->\n\nCommon symptoms include increased thirst, frequent urination, and fatigue.",
            metadata={"term": "Diabetes", "section": "Symptoms", "page": 5}
        )
    """
    documents = []

    for item in structured_data:
        term = item["term"]
        section = item["section"]
        content = item["content"]
        page = item["page"]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        chunks = text_splitter.split_text(content)
        
        for chunk in chunks:
            # Format text with term and section as headers
            formatted_text = f"Page: {page} -> {term} -> {section} ->\n\n{chunk}"
            
            documents.append(
                Document(
                    page_content=formatted_text,
                    metadata={"term": term, "section": section, "page": page},
                )
            )

    return documents

def download_hugging_face_embeddings() -> HuggingFaceEmbeddings:
    """
    Download a Hugging Face embeddings model.

    The function downloads a Hugging Face embeddings model with the model name 'sentence-transformers/all-MiniLM-L6-v2' and returns a HuggingFaceEmbeddings object.

    Returns:
        HuggingFaceEmbeddings: A HuggingFaceEmbeddings object containing the downloaded model.
    """
    embeddings=HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
        )  #this model return 384 dimensions
    return embeddings