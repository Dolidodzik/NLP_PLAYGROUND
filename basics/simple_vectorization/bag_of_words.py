import spacy

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
print("\n ================================ \n")
