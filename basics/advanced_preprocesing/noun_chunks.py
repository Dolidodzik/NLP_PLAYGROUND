import spacy
nlp = spacy.load("en_core_web_sm")

text = "The quick brown fox jumped over the lazy dog."
doc = nlp(text)

for chunk in doc.noun_chunks:
    print(chunk.text)
