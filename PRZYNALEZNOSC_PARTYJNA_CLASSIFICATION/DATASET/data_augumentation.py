import random

import pandas as pd
import spacy


nlp = spacy.load('pl_core_news_sm')
pl_stop = nlp.Defaults.stop_words
from nltk.tokenize import word_tokenize


dataset_df = pd.read_csv('TRAINING_DATASET_RAW_2024_TO_MAY_2025.csv')
print(dataset_df)

example_speech = '''
wysoka izbo, ta ustawa to jest absolutne minimum bezpieczeństwa i to jest ustawa, za którą naprawdę może zagłosować każdy uczciwy konserwatysta, który będzie się z nami, z lewicą spierać o wartości, o rozwiązania, o praktykę. tutaj mówimy po prostu o bezpieczeństwie. tutaj mówimy o rozwiązaniu, które ma skończyć z sytuacją, w której polskie państwo używa narzędzi karnych, żeby ścigać swoich własnych obywateli w sytuacji, kiedy tego robić po prostu nie powinno. mój apel jest taki: drodzy państwo, nie zacinajcie się, spójrzcie na tekst tej ustawy i zagłosujcie rozumnie, po prostu zagłosujcie za tą ustawą. dziękuję.
'''

# synonym replacement
def synonym_replace_for_sentence(sentence, frac=0.5):
    tokens = word_tokenize(sentence, language='polish')
    candidates = [i for i, w in enumerate(tokens) if w.lower() not in pl_stop]
    random.shuffle(candidates)
    n_to_replace = max(1, int(len(candidates) * frac))
    replaced = 0
    for idx in candidates:
        if replaced >= n_to_replace:
            break
        # ... fetch synonyms with plWordnet and replace tokens[idx] ...
        replaced += 1
    return ' '.join(tokens)

print(synonym_replace_for_sentence("wysoka izbo, ta ustawa to jest absolutne minimum bezpieczeństwa."))


# back translation

# contextual masking