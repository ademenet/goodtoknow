# @Author: Alain
# @Date:   2016-09-17T17:01:42+02:00
# @Last modified by:   Alain
# @Last modified time: 2016-09-19T11:46:07+02:00

import json
import re
import pandas as pd
from bs4 import BeautifulSoup
from pprint import pprint
import nltk
from nltk.corpus import stopwords

json_file = 'data_film_json'
json_data = open(json_file)
data = json.load(json_data)

def parsing_and_cleaning(raw_text):
	# Function to convert a raw text into clean lower-case words.
	# Input: string
	# Output: string of clean words only

	# Clean HTML tags
	text = BeautifulSoup(raw_text, "lxml").get_text()
	# Clean non-letters
	letters_only = re.sub("[^a-zA-Z]", " ", text)
	# Convert to lower-case and split to words
	words = letters_only.lower().split()
	# Convert stop words into a set rather than list because it's much faster
	stops = set(stopwords.words("english"))
	# Then remove stop words
	final_words = [w for w in words if not w in stops]
	# Join the words into one string separated by spaces
	return(" ".join(final_words))

# Initialize our list and choose a category
texts = []
tests = []
category = []
cat = 'Comedy'

i = 0

# Iterating into datas
for film in data:
	if 'all_genres_ss' in film:
		for genre in film['all_genres_ss']:
			if genre == 'Comedy' or genre == 'Drama':
				if 't_synopsis_text_en' in film:
					category.append(genre)
					words = parsing_and_cleaning(film['t_synopsis_text_en'][0])
					texts.append(words)

					if i < 201:
						tests.append(genre)
					i += 1


# Slice texts in training datas and tests
print len(texts)
data_set_test = texts[:201]
data_set_train = texts[-1000:]

# Creating the bag of words using scikit-learn
from sklearn.feature_extraction.text import CountVectorizer
# Vectorizer is the scikit's bag of words object
vectorizer = CountVectorizer(analyzer = "word",\
                             tokenizer = None,\
                             preprocessor = None,\
                             stop_words = None,\
                             max_features = 5000)

train_data_features = vectorizer.fit_transform(data_set_train)
train_data_features = train_data_features.toarray()

# Let's try random forest classifier here
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data_features, category[-1000:])

# Tests !
test_data_features = vectorizer.transform(data_set_test)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)

import numpy as np
from sklearn.metrics import accuracy_score
print accuracy_score(tests, result)

output = pd.DataFrame(data = { "predictions":result, "original":tests })
output.to_csv("resultats.csv", index=False, quoting=3)

# Another classifier method:
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(train_data_features, category[-1000:])
