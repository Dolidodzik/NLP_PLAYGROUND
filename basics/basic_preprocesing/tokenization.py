import spacy
from spacy.util import compile_infix_regex
import re

nlp = spacy.load("pl_core_news_sm")

#hyphen handling
infixes = [
    pattern for pattern in nlp.Defaults.infixes 
    if not re.search(r'[-–—]', pattern)  # remove hyphens from splitting rules
]
infix_regex = compile_infix_regex(infixes)
nlp.tokenizer.infix_finditer = infix_regex.finditer

#add abbreviations
abbreviations = ['np.', 'tzw.', 'itd.']
for abbrev in abbreviations:
    nlp.tokenizer.add_special_case(abbrev, [{"ORTH": abbrev}])

# processing text after tokenizer nlp thing was set up
text = 'Tzw. "Polska" prawica np. slawomir memcen to jawni oszusci XD. Mieszkam w Bielsku-Białej i widzę co sie dzieje na ulicach, przestępczość itd.'
text = text.lower()

doc = nlp(text)

lemmatized_tokens = [token.lemma_ for token in doc]

print(lemmatized_tokens)