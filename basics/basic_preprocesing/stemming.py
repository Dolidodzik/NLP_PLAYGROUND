from nltk.stem import PorterStemmer # porter steamer make it ['run', 'jump', 'happili', 'cat']
stemmer = PorterStemmer()

# Lancaster Stemmer is more aggresive - ['run', 'jump', 'happy', 'cat']
#from nltk.stem import LancasterStemmer
#stemmer = LancasterStemmer()

words = ["running", "jumps", "happily", "cats"]
stemmed = [stemmer.stem(word) for word in words]

print(stemmed)