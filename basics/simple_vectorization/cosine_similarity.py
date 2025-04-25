import spacy
from math import sqrt
import numpy as np
from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = [
  "red red dog car car test",
  "red dog car test",
  "nothing empty void pest",
  "dog red car car test"
]

vectorizer = CountVectorizer()
bow = vectorizer.fit_transform(corpus)
print("vectorizer.vocabulary_: ", vectorizer.vocabulary_)
print(bow.toarray())
print("\n ================================= \n")

doc1_vs_doc2 = 1 - spatial.distance.cosine(bow[0].toarray()[0], bow[1].toarray()[0])
doc1_vs_doc3 = 1 - spatial.distance.cosine(bow[0].toarray()[0], bow[2].toarray()[0])
doc1_vs_doc4 = 1 - spatial.distance.cosine(bow[0].toarray()[0], bow[3].toarray()[0])

print("scipy spatial cosine similarity: ")
print(f"Doc 1 vs Doc 2: {doc1_vs_doc2}")
print(f"Doc 1 vs Doc 3: {doc1_vs_doc3}")
print(f"Doc 1 vs Doc 4: {doc1_vs_doc4}")

print("=============================")
print("my own cosine similarity implentation (without numpy)")
print("=============================")

def compute_cosine_similarity(array1, array2):
    if len(array1) != len(array2):
        raise ValueError("ARRAY LENGTHS NOT MATCH")

    dot_product = 0
    array1_sum_of_squares = 0
    array2_sum_of_squares = 0
    for i in range(0, len(array1)):
        dot_product += array1[i] * array2[i]
        array1_sum_of_squares += array1[i] * array1[i]
        array2_sum_of_squares += array2[i] * array2[i]
    
    denominator = sqrt(array1_sum_of_squares) * sqrt(array2_sum_of_squares)
    return dot_product / denominator

doc1_vs_doc2 = compute_cosine_similarity(bow[0].toarray()[0], bow[1].toarray()[0])
doc1_vs_doc3 = compute_cosine_similarity(bow[0].toarray()[0], bow[2].toarray()[0])
doc1_vs_doc4 = compute_cosine_similarity(bow[0].toarray()[0], bow[3].toarray()[0])

print("my own cosine similarity (without numpy): ")
print(f"Doc 1 vs Doc 2: {doc1_vs_doc2}")
print(f"Doc 1 vs Doc 3: {doc1_vs_doc3}")
print(f"Doc 1 vs Doc 4: {doc1_vs_doc4}")


print("=============================")
print("my own cosine similarity implentation with numpy")
print("=============================")

def compute_cosine_similarity_numpy(array1, array2):
    a1 = np.asarray(array1)
    a2 = np.asarray(array2)
    
    if a1.shape != a2.shape:
        raise ValueError("Array shapes do not match")
    
    print(a1.shape)
    
    dot_product = np.dot(a1, a2)
    
    norm_a1 = np.linalg.norm(a1)
    norm_a2 = np.linalg.norm(a2)
    
    if norm_a1 == 0 or norm_a2 == 0:
        return 0.0
    
    return dot_product / (norm_a1 * norm_a2)

doc1_vs_doc2 = compute_cosine_similarity_numpy(bow[0].toarray()[0], bow[1].toarray()[0])
doc1_vs_doc3 = compute_cosine_similarity_numpy(bow[0].toarray()[0], bow[2].toarray()[0])
doc1_vs_doc4 = compute_cosine_similarity_numpy(bow[0].toarray()[0], bow[3].toarray()[0])

print("my own cosine similarity (with numpy): ")
print(f"Doc 1 vs Doc 2: {doc1_vs_doc2}")
print(f"Doc 1 vs Doc 3: {doc1_vs_doc3}")
print(f"Doc 1 vs Doc 4: {doc1_vs_doc4}")