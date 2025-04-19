import spacy

nlp = spacy.load("en_core_web_sm")

text = "SpaCy is an amazing library." # hierarchical piramid that all comes to "is" in this case
doc = nlp(text)

for token in doc:
    print(f'{token.text} --> {token.dep_} --> {token.head.text}')
