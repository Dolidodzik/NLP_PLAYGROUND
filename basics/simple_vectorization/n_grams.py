import spacy
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.load('en_core_web_sm')

def spacy_tokenizer(text):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc 
            if not token.is_stop and not token.is_punct]

corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "Never jump over lazy dogs quickly.",
    "Quick foxes and lazy dogs.",
    "The dog is lazy but the fox is quick."
]

vectorizer = CountVectorizer(
    tokenizer=spacy_tokenizer,
    binary=True,
    ngram_range=(3, 3)
)
'''
ngram_range is (1,1) by default so only unigrams like "fox", "lazy", "dogs" are generated
ngram_range (2,2) generates only bigrams like "jump lazy", "dog quickly"
ngram_range (3,3) generates only trigrams like "fox jump lazy", "lazy dog quickly"

ngram_range (2,3) generates both bigrams and trigrams
ngram_range (1,3) generates unigrams and bigrams and trigrams
'''

bigram_matrix = vectorizer.fit_transform(corpus)

print("=== Feature Names (Vocabulary) ===")
print(vectorizer.get_feature_names_out())
print(f'\nNumber of features: {len(vectorizer.get_feature_names_out())}')

print("\n=== Vocabulary Mapping ===")
print(vectorizer.vocabulary_)

print("\n=== Bigram Matrix (Document-Term Matrix) ===")
print(bigram_matrix.toarray())