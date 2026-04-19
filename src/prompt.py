import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def get_metadata_prompt() -> str:
    """
    Returns prompt for extracting term and section metadata 
    """
    section_set = []
    term_set = []

    with open(os.path.join(BASE_DIR, "section_set.txt"), "r") as f:
        section_set = [line.strip() for line in f.readlines()]
    with open(os.path.join(BASE_DIR, "term_set.txt"), "r") as f:
        term_set = [line.strip() for line in f.readlines()]

    # Metadata Prompt
    METADATA_PROMPT = f"""
    Extract structured medical query information.

    Rules:
    - Extract section names ONLY from this list:
    {section_set}

    - Extract the closest matching medical term from this list:
    {term_set}

    Return output in JSON format with keys:
    term: string
    section: list[string]
    """

    return str(METADATA_PROMPT)

def get_system_prompt() -> str:
    # System Prompt
    """
    Returns a prompt for generating system responses.

    The system prompt is a string that gives instructions to the model on how to generate a response.
    """
    SYSTEM_PROMPT = (
    "You are an Medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use five sentences maximum and keep the "
    "answer concise. Mention section and page number from the book along with the answer"
    "like this (Causes and Symptoms,p 24)."
    " "
)


    return str(SYSTEM_PROMPT)

if __name__ == "__main__":
    print(get_metadata_prompt())
    print(get_system_prompt())