import nltk
from nltk.stem import PorterStemmer
import json
import numpy as np

# Download NLTK resources
nltk.download('punkt')

# Define function to stem words
def get_stem_words(words_to_stem, ignore_words):
    porter_stemmer = PorterStemmer()
    stem_words = []
    for word in words_to_stem:
        if word not in ignore_words:
            stemmed_word = porter_stemmer.stem(word.lower())
            stem_words.append(stemmed_word)
    return stem_words

# Load data from JSON file
with open('intents.json') as file:
    data = json.load(file)

words = []  # List of unique root words in the data
classes = []  # List of unique tags in the data
patterns_tags_list = []  # List of pairs of (['words', 'of', 'the', 'sentence'], 'tags')
ignore_words = ['?', '!', ',', '.', "'s", "'m"]  # Words to be ignored

# Iterate through the intents in the JSON data
for intent in data['intents']:
    # Iterate through the patterns and tags in each intent
    for pattern in intent['patterns']:
        # Tokenize the pattern
        pattern_words = nltk.word_tokenize(pattern)
        # Add tokenized words to the words list
        words.extend(pattern_words)
        # Add pair of pattern words and tag to patterns_tags_list
        patterns_tags_list.append((pattern_words, intent['tag']))
    # Add unique tag to classes list
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Stem the words
stem_words = get_stem_words(words, ignore_words)

# Remove duplicate words and sort the stem_words list
stem_words = sorted(list(set(stem_words)))
classes = sorted(list(set(classes)))

print("Stemmed words:", stem_words)
print("Classes:", classes)
print("Patterns and tags list:", patterns_tags_list)