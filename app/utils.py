import re

def validate_query(query: str) -> str:
    """
    Validate user query:
    - Must be string and not empty
    - Cannot be only whitespace/punctuation
    - Max 500 words
    - Reject suspicious patterns (basic prompt injection guard)
    """

    if not isinstance(query, str):
        return "Input tidak valid. Harap masukkan teks pertanyaan."

    query = query.strip()
    if not query or re.fullmatch(r"[^\w]+", query):
        return "Silakan masukkan pertanyaan yang valid."

    if len(query.split()) > 500:
        return "Pertanyaan terlalu panjang. Coba ringkas kembali."

    forbidden = ["abaikan", "abaikan instruksi", "ignore previous bypass", 
                 "sistem prompt", "change rules"]
    if any(p in query.lower() for p in forbidden):
        return "Pertanyaan tidak sesuai aturan keamanan."

    return query


def clean_query(text):
    """
    Normalize and clean user input text.

    Steps:
    1. Convert to lowercase
    2. Trim leading and trailing spaces
    3. Replace multiple spaces with a single space
    4. Normalize repeated punctuation marks (e.g., '!!!' -> '!')
    5. Remove non-alphanumeric characters (except space, ., ?, !)

    :param text: The input string from the user
    :return: A cleaned and normalized string
    """
    text = text.lower()
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([!?])\1+", r"\1", text)
    text = re.sub(r"[^a-z0-9\s\.\?\!]", "", text)
    return text
