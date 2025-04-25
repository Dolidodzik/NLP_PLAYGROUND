import spacy
import pandas as pd
from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = [
  "Red Bull drops hint on F1 engine.",
  "Honda exits F1, leaving F1 partner Red Bull.",
  "Hamilton eyes record eighth F1 title.",
  "Aston Martin announces sponsor."
]

vectorizer = CountVectorizer()

bow = vectorizer.fit_transform(corpus)
print("feature names out: ", vectorizer.get_feature_names_out())
print("vectorizer.vocabulary_: ", vectorizer.vocabulary_)
print("type of bow variable: ", (type(bow)))
print("raw bow: ", bow)

df_bow = pd.DataFrame(bow.toarray(), columns=vectorizer.get_feature_names_out())
print("bow as pandas data frame: \n")
print(df_bow)
df_bow.to_csv("BOW.csv", index=False)


print("\n ================================ \n")


nlp = spacy.load('en_core_web_sm')
def spacy_tokenizer(doc):
  return [t.text for t in nlp(doc) if not t.is_punct]

vectorizer = CountVectorizer(tokenizer=spacy_tokenizer, lowercase=False, binary=True)
bow = vectorizer.fit_transform(corpus)

print("feature names out: ", vectorizer.get_feature_names_out())
print("vectorizer.vocabulary_: ", vectorizer.vocabulary_)
print("==============================")
print('A dense representation like we saw in the slides.')
print(bow.toarray())
print()
print('Indexing and slicing.')
print(bow[0])
print()
print(bow[0:2])
