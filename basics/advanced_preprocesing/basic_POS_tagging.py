import spacy
nlp = spacy.load("pl_core_news_sm")

doc = nlp("Volkswagen rozwija elektryczny sedan.")
tokens = [(token.text, token.pos_) for token in doc]

print("normal POS tagging: ")
print(tokens)
print(spacy.explain('PROPN'))

print("=================================")

print("FINE GRAINED POS TAGGING, more detailed more specific info")
tokens = [(token.text, token.tag_) for token in doc]
print(tokens)