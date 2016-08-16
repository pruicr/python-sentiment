import re, math, collections, itertools, os
import nltk, nltk.classify.util, nltk.metrics
from nltk import precision, recall
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

POLARITY_DATA_DIR = os.path.join('polarityData', 'rt-polaritydata')
RT_POLARITY_POS_FILE = os.path.join(POLARITY_DATA_DIR, 'rt-polarity-pos.txt')
RT_POLARITY_NEG_FILE = os.path.join(POLARITY_DATA_DIR, 'rt-polarity-neg.txt')

def evaluate_features(feature_selection):
    posFeatures = []
    negFeatures = []

    with open(RT_POLARITY_POS_FILE, 'r') as posSentence:
        for i in posSentence:
            posWords = re.findall(r"[\w']+|[.,!?;]", i.strip())
            posWords = [feature_selection(posWords), 'pos']
            posFeatures.append(posWords)
    with open(RT_POLARITY_NEG_FILE, 'r') as negSentence:
        for i in negSentence:
            negWords = re.findall(r"[\w']+|[.,!?:]", i.strip())
            negWords = [feature_selection(negWords), 'neg']
            negFeatures.append(negWords)

    posCutoff = int(math.floor(len(posFeatures)*3/4))
    negCutoff = int(math.floor(len(negFeatures)*3/4))
    trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
    testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]

    classifier = NaiveBayesClassifier.train(trainFeatures)

    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set)

    for i, (features, label) in enumerate(testFeatures):
        referenceSets[label].add(i)
        predicted = classifier.classify(features)
        testSets[predicted].add(i)

    print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
    print 'accuracy:', nltk.classify.util.accuracy(classifier,testFeatures)
    print 'pos precision:', precision(referenceSets['pos'], testSets['pos'])
    print 'pos recall:', recall(referenceSets['pos'], testSets['pos'])
    print 'neg precision:', precision(referenceSets['neg'], testSets['neg'])
    print 'neg recall:', recall(referenceSets['neg'], testSets['neg'])
    classifier.show_most_informative_features(10)

def make_full_dict(words):
    return dict([(word, True) for word in words])

print 'using all words as features'
evaluate_features(make_full_dict)

def create_word_scores():

    posWords = []
    negWords = []
    with open(RT_POLARITY_POS_FILE, 'r') as posSentence:
        for i in posSentence:
            posWord = re.findall(r"[\w']+|[.,!?;]", i.strip())
            posWords.append(posWord)
    with open(RT_POLARITY_NEG_FILE, 'r') as negSentence:
        for i in negSentence:
            negWord = re.findall(r"[\w']+|[.,!?;]", i.strip())
            negWords.append(negWord)
    posWords = list(itertools.chain(*posWords))
    negWords = list(itertools.chain(*negWords))

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in posWords:
        word_fd[word.lower()] += 1
        cond_word_fd['pos'][word.lower()] += 1
    for word in negWords:
        word_fd[word.lower()] += 1
        cond_word_fd['neg'][word.lower()] += 1

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores

word_scores = create_word_scores()

def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words

def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])

numbers_to_test = [10, 100, 1000, 10000, 15000]

for num in numbers_to_test:
    print 'evaluating best %d word features' % num
    best_words = find_best_words(word_scores, num)
    evaluate_features(best_word_features)
