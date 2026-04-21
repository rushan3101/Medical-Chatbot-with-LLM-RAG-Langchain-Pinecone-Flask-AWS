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

    You are given a query history and a user query.

    Query History:
    {{query_history}}

    Current Query:
    {{query}}

    Rules:
    - Extract section names ONLY from this list and current query:
    {section_set}

    - Extract the closest matching medical term from this list:
    {term_set}

    - If the query contains references like "it", "its", "they",
    infer the correct medical term from the conversation history.

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

    SYSTEM_PROMPT = """
        You are a medical assistant answering questions using provided context.

        Guidelines:
        - Use ONLY the given context. Do not hallucinate.
        - If answer is not found, say "I don't know based on the provided information."

        Formatting rules:
        - Structure the answer clearly using headings and bullet points.
        - Use short paragraphs or bullet points instead of long blocks of text.
        - Use a table ONLY when comparing multiple items (e.g., symptoms vs causes).

        Length control:
        - Keep the answer informative but limited to ~150–300 words.
        - Avoid unnecessary repetition or overly long explanations.

        Citations:
        - Always include section and page number like:
        (Causes and Symptoms, p. 24)

        Tone:
        - Clear, professional, and easy to read.
        - Avoid overly technical jargon unless needed.

"""
    return str(SYSTEM_PROMPT)