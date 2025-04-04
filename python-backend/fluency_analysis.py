import sys, json
import spacy
from collections import Counter

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")


def analyze_speech_fluency(text):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
    total_tokens = len(tokens)
    unique_tokens = set(tokens)
    ttr = len(unique_tokens) / total_tokens if total_tokens > 0 else 0

    filler_words = {"um", "uh", "like", "you know", "so", "actually", "basically", "i mean"}
    filler_count = sum(1 for token in tokens if token in filler_words)

    repeated_tokens = sum(1 for token, count in Counter(tokens).items() if count > 1)

    return {
        "total_tokens": total_tokens,
        "unique_tokens": len(unique_tokens),
        "type_token_ratio": ttr,
        "filler_count": filler_count,
        "repeated_tokens": repeated_tokens,
    }


if __name__ == "__main__":
    input_data = sys.stdin.read()
    request = json.loads(input_data)
    text = request.get("text", "")
    result = analyze_speech_fluency(text)
    print(json.dumps(result))
    sys.stdout.flush()  # Ensure the output is flushed
