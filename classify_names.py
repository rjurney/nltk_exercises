import nltk
from nltk.corpus import names
import random

def gender_features(word):
  vowels = "aeiuoAEIOU"
  return {'last_letter': word[-1], 'length': len(word), 'first_letter': word[0].lower(), 'vowel_count': len([letter for letter in word if letter in vowels])}

names = [(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')]
random.shuffle(names)

featureset = [(gender_features(n), g) for (n, g) in names]
train_set, test_set = featureset[500:], featureset[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print nltk.classify.accuracy(classifier, test_set)
classifier.show_most_informative_features()
