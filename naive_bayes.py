from decimal import *
from functools import reduce

poslines = [line.strip() for line in open('pos.train')]
neglines = [line.strip() for line in open('neg.train')]

poswords = []
negwords = []

for l in poslines:
    poswords.extend(l.split())

for l in neglines:
    negwords.extend(l.split())

#This function uses the training data to get counts of all words in each train file
def train(words):
    word_count = dict()
    for word in words:
            if word in word_count:
                word_count[word] = word_count[word] + 1
            else:
                word_count[word] = 1

    return word_count

pos_count = train(poswords)
neg_count = train(negwords)

vocabulary = list(set(pos_count.keys()).union(set(neg_count.keys())))

#This function smooths the data with .5
def smooth(vocab, count):
    smoothed_count = dict()
    for word in vocab:
        if word in count:
            smoothed_count[word] = count[word] + .5
        else:
            smoothed_count[word] = .5

    return smoothed_count

new_pos_count = smooth(vocabulary, pos_count)
new_neg_count = smooth(vocabulary, neg_count)

#This function gets the probabilty for a given word
def get_probability(word, count):
    total_count = sum(count.values())
    if word in count :
        return Decimal(count[word]/total_count)
    else:
        #I am returning 1 here so that if the word is unknown the probability calcuation will not be affected.
        # this essentially ignores any unknown words.
        return 1


#This function returns 1 if the review is classified as positive and 0 if it is classified as negative.
# We do not need to include the prior in  our calculations becausee it is the  same for both positive and negative reviews.
def classifier(review):
    words = review.split()
    pos_probabilities = [get_probability(word, new_pos_count) for word in words]
    neg_probabilities = [get_probability(word, new_neg_count) for word in words]
    total_pos_probability = reduce((lambda x, y: x * y), pos_probabilities)
    total_neg_probability = reduce((lambda x, y: x * y), neg_probabilities)
    return 1 if total_pos_probability > total_neg_probability else 0

print(classifier("this is great"))
print(classifier("this is awful"))

import os

pos_filenames = [f for f in os.listdir('test/pos') if not f.startswith('.')]
neg_filenames = [f for f in os.listdir('test/neg') if not f.startswith('.')]


def evaluate(filenames, path, desired_output):
    accurate_classifier_count = 0
    for f in filenames:
        lines = [line.strip() for line in open(path + f)]
        if classifier("".join(lines)) == desired_output:
            accurate_classifier_count += 1
    return accurate_classifier_count

print(evaluate(pos_filenames, 'test/pos/', 1)) #81% accuate for positive reviews
print(evaluate(neg_filenames, 'test/neg/', 0)) #86% accurate for negative reviews
#This means the overall accuracy is 167/200 or 83.5% accuracy
