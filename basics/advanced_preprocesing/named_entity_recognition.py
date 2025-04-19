import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

s = "Volkswagen is developing an electric sedan which could potentially come to America next fall. And by the way, we'll be in Osaka on Feb 13th and leave on Feb 24th."
s = s.lower()
doc = nlp(s)
tokens = [(token.text, token.ent_type_) for token in doc]

print("basic example: ")
print(tokens)

print("============================")
print("extracting only dates")

dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
print(type(dates[0]))
print(dates)
