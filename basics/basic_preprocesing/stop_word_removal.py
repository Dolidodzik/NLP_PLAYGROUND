import spacy
nlp = spacy.load('en_core_web_sm')

def remove_stop_words(text):
    doc = nlp(text)
    return_list = []

    for token in doc:
        print("taking a look at token: ", token)
        print(type(token))
        if not token.is_stop:
            return_list.append(token.text)
            print("appended following text: ", token.text)
            print(type(token.text))
        print("=============================")

    return return_list

text = "Let's go to N.Y.C. for the weekend."

# additional case folding
text = text.lower()

filtered_tokens = remove_stop_words(text)
print("="*50)
print(filtered_tokens)