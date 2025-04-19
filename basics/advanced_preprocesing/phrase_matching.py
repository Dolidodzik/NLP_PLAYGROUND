import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")
text = "I love my iPhone 12. The Samsung Galaxy is also a good phone."
text = text.lower()
doc = nlp(text)

matcher = PhraseMatcher(nlp.vocab)

patterns = [nlp(text) for text in ["iphone 12", "samsung galaxy"]]
print("patterns: ", patterns)
for pattern in patterns:
    matcher.add("PRODUCT", None, pattern)

matches = matcher(doc)
print("matches: ", matches)

for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(matched_span.text)
