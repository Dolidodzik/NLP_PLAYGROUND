import spacy

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



corpus = fetch_20newsgroups(categories=['sci.space'],
                            remove=('headers', 'footers', 'quotes'))

print("corpus example entry")
print(corpus.data[0])
print(corpus.target_names[0])
print("corpus length: ", len(corpus.data))

nlp = spacy.load('en_core_web_sm')
unwanted_pipes = ["ner", "parser"]

def spacy_tokenizer(doc):
  with nlp.disable_pipes(*unwanted_pipes):
    return [t.lemma_ for t in nlp(doc) if \
            not t.is_punct and \
            not t.is_space and \
            t.is_alpha]

vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 2))
features = vectorizer.fit_transform(corpus.data)

# The number of unique tokens.
print(len(vectorizer.get_feature_names_out()))

print("features shape: ", features.shape)
print("first feature: ", features[0])


query = ["Bill Gates"]
query_tfidf = vectorizer.transform(query)
print("querying for: ", str(query))
print("query shape: ", query_tfidf.shape)

cosine_similarities = cosine_similarity(features, query_tfidf).flatten()
print("shape of cosine similarties: ", cosine_similarities.shape)

# list of indexes in cosine_similarities. Sorted based on cosine_similarities values.  cosine_similarities[ascending_idx[0]] is the smallest value in cosine_similarities, and so on
ascending_idx = np.argsort(cosine_similarities)

descending_idx = ascending_idx[::-1]
for i in range(5):
    doc_index = descending_idx[i]
    similarity = cosine_similarities[doc_index]
    doc_text = corpus.data[doc_index]
    
    print(f"Doc #{doc_index} | Similarity: {similarity:.4f}")
    print("-" * 60)
    print(doc_text)  # show first 500 chars (you can change or remove this)
    print("=" * 60)