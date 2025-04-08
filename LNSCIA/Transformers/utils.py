import re
import unicodedata

def normalize_text(text):
    text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z0-9\\s]", "", text)
    return text
