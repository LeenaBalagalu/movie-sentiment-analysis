'''
  This program shell reads phrase data for the kaggle phrase sentiment classification problem.
  The input to the program is the path to the kaggle directory "corpus" and a limit number.
  The program reads all of the kaggle phrases, and then picks a random selection of the limit number.
  It creates a "phrasedocs" variable with a list of phrases consisting of a pair
    with the list of tokenized words from the phrase and the label number from 1 to 4
  It prints a few example phrases.
  In comments, it is shown how to get word lists from the two sentiment lexicons:
      subjectivity and LIWC, if you want to use them in your features
  Your task is to generate features sets and train and test a classifier.

  Usage:  python classifyKaggle.py  <corpus directory path> <limit number>

  This version uses cross-validation with the Naive Bayes classifier in NLTK.
  It computes the evaluation measures of precision, recall and F1 measure for each fold.
  It also averages across folds and across labels.
'''
# open python and nltk packages needed for processing
import os
import sys
import random
import nltk
from nltk.corpus import stopwords
from string import punctuation
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import bigrams
import string
import re
from collections import Counter
from nltk.util import bigrams
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy.sparse import vstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier


# Set the random state for reproducibility
random.seed(42)

## this code is commented off now, but can be used for sentiment lists
import sentiment_read_subjectivity
# initialize the positive, neutral and negative word lists
(positivelist, neutrallist, negativelist) = sentiment_read_subjectivity.read_subjectivity_three_types('SentimentLexicons/subjclueslen1-HLTEMNLP05.tff')

import sentiment_read_LIWC_pos_neg_words
# initialize positve and negative word prefix lists from LIWC 
#   note there is another function isPresent to test if a word's prefix is in the list
(poslist, neglist) = sentiment_read_LIWC_pos_neg_words.read_words()

## define a feature definition function here

# this function define features (keywords) of a document for a BOW/unigram baseline
# each feature is 'V_(keyword)' and is true or false depending
# on whether that keyword is in the document

# UNIGRAM BAG OF WORDS FEATURE
def unigram_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    return features

# UNIGRAM AND NEGATION FEATURE
def unigram_and_negation_features(document, word_features):
    document_words = set(document)
    features = {}
    # Unigram Features
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    # Negation Features
    negation_words = {"not", "no", "never", "none", "nothing", "nowhere", "neither", "nor",
        "noone", "rather", "hardly", "scarcely", "rarely","seldom"}
    negated_context = False
    for i, word in enumerate(document):
        if word in negation_words:
            negated_context = True
        elif negated_context:
            # Add negation feature for the word
            features['V_Neg_{}'.format(word)] = True
            # Negation scope is limited to 3 words
            if i > 0 and i % 3 == 0:
                negated_context = False
    return features
    
#POS TAG FEATURE
def pos_tag_features(document, word_features):
    pos_tags = [tag for word, tag in nltk.pos_tag(document) if word in word_features]
    pos_counts = Counter(pos_tags)  # Count each POS tag
    features = {}
    for tag, count in pos_counts.items():
        features[f'POS_{tag}'] = count
    return features



# SUBJECTIVITY LEXICON
def subjectivity_features(document, positivelist, neutrallist, negativelist):
    features = {}
    document_words = set(document)

    # Count occurrences of words in each category
    positive_count = sum(1 for word in document_words if word in positivelist)
    neutral_count = sum(1 for word in document_words if word in neutrallist)
    negative_count = sum(1 for word in document_words if word in negativelist)

    # Add counts to features
    features["positive_count"] = positive_count
    features["neutral_count"] = neutral_count
    features["negative_count"] = negative_count

    return features
    
# LIWC LEXICON FEATURE
def liwc_features(document,poslist, neglist):
    features = {}
    positive_count = 0
    negative_count = 0  
    # Count occurrences of LIWC positive and negative words in the document
    for word in document:
        if sentiment_read_LIWC_pos_neg_words.isPresent(word, poslist):
            positive_count += 1
        if sentiment_read_LIWC_pos_neg_words.isPresent(word, neglist):
            negative_count += 1   
    # Add counts as features
    features['LIWC_Positive'] = positive_count
    features['LIWC_Negative'] = negative_count
    return features
    
# VADER LEXICON FEATURE
sia = SentimentIntensityAnalyzer()

def vader_features(document):
    # Join tokens to form a single string (VADER expects full sentences)
    text = ' '.join(document)   
    # Get sentiment scores from VADER
    sentiment_scores = sia.polarity_scores(text)    
    # Add sentiment scores as features
    features = {
        'VADER_Positive': sentiment_scores['pos'],  # Positive sentiment score
        'VADER_Neutral': sentiment_scores['neu'],   # Neutral sentiment score
        'VADER_Negative': sentiment_scores['neg'],  # Negative sentiment score
        'VADER_Compound': sentiment_scores['compound']  # Compound sentiment score
    }
    return features

# BIGRAM FEATURE
def bigram_features(document, bigram_features_list):
    document_bigrams = set(bigrams(document))  # Generate bigrams from the document
    features = {}
    for bigram in bigram_features_list:
        features[f'B_{bigram[0]}_{bigram[1]}'] = (bigram in document_bigrams)
    return features
 
# COMBINED FEATURE 

def combined_features(document, word_features,
                      positivelist, neutrallist, negativelist,
                      poslist, neglist):
    features = {}
    features.update(unigram_features(document, word_features))         # Unigram and negation features
    #features.update(bigram_features(document, bigram_features_list))   # Bigram features
    features.update(pos_tag_features(document, word_features))                        # POS tag features
    features.update(subjectivity_features(document, positivelist, neutrallist, negativelist))  # Subjectivity features
    features.update(liwc_features(document, poslist, neglist))         # LIWC features
    features.update(vader_features(document))           # VADER features
    return features


## cross-validation ##
# this function takes the number of folds, the feature sets and the labels
# it iterates over the folds, using different sections for training and testing in turn
#   it prints the performance for each fold and the average performance at the end

def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    return features

## cross-validation ##
# this function takes the number of folds, the feature sets and the labels
# it iterates over the folds, using different sections for training and testing in turn
#   it prints the performance for each fold and the average performance at the end
## cross-validation ##
def cross_validation_PRF(num_folds, featuresets, labels):
    subset_size = int(len(featuresets) / num_folds)
    print('Each fold size:', subset_size)
    # for the number of labels - start the totals lists with zeroes
    num_labels = len(labels)
    total_precision_list = [0] * num_labels
    total_recall_list = [0] * num_labels
    total_F1_list = [0] * num_labels
    total_accuracy = 0  # Initialize accuracy accumulator

    # iterate over the folds
    for i in range(num_folds):
        test_this_round = featuresets[(i * subset_size):][:subset_size]
        train_this_round = featuresets[:(i * subset_size)] + featuresets[((i + 1) * subset_size):]
        # train using train_this_round
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        # evaluate against test_this_round to produce the gold and predicted labels
        goldlist = []
        predictedlist = []
        for (features, label) in test_this_round:
            goldlist.append(label)
            predictedlist.append(classifier.classify(features))

        # computes evaluation measures for this fold and
        #   returns list of measures for each label
        #print('Fold', i)
        (precision_list, recall_list, F1_list) \
                  = eval_measures(goldlist, predictedlist, labels)

        # Compute accuracy for this fold
        correct_predictions = sum(1 for gold, pred in zip(goldlist, predictedlist) if gold == pred)
        accuracy = correct_predictions / len(goldlist)
        total_accuracy += accuracy  # Accumulate accuracy

        # for each label add to the sums in the total lists
        for j in range(num_labels):
            # for each label, add the 3 measures to the 3 lists of totals
            total_precision_list[j] += precision_list[j]
            total_recall_list[j] += recall_list[j]
            total_F1_list[j] += F1_list[j]

    # find precision, recall and F measure averaged over all rounds for all labels
    # compute averages from the totals lists
    precision_list = [tot / num_folds for tot in total_precision_list]
    recall_list = [tot / num_folds for tot in total_recall_list]
    F1_list = [tot / num_folds for tot in total_F1_list]
    average_accuracy = total_accuracy / num_folds  # Average accuracy

    # the evaluation measures in a table with one row per label
    print('\nAverage Precision\tRecall\t\tF1 \tPer Label')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
          "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))
    
    # print macro average over all labels - treats each label equally
    print('\nMacro Average Precision\tRecall\t\tF1 \tOver All Labels')
    print('\t', "{:10.3f}".format(sum(precision_list) / num_labels), \
          "{:10.3f}".format(sum(recall_list) / num_labels), \
          "{:10.3f}".format(sum(F1_list) / num_labels))

    # for micro averaging, weight the scores for each label by the number of items
    #    this is better for labels with imbalance
    # first initialize a dictionary for label counts and then count them
    label_counts = {}
    for lab in labels:
      label_counts[lab] = 0 
    # count the labels
    for (doc, lab) in featuresets:
      label_counts[lab] += 1
    # make weights compared to the number of documents in featuresets
    num_docs = len(featuresets)
    label_weights = [(label_counts[lab] / num_docs) for lab in labels]
    print('\nLabel Counts', label_counts)
    #print('Label weights', label_weights)
    # print macro average over all labels
    print('Micro Average Precision\tRecall\t\tF1 \tOver All Labels')
    precision = sum([a * b for a, b in zip(precision_list, label_weights)])
    recall = sum([a * b for a, b in zip(recall_list, label_weights)])
    F1 = sum([a * b for a, b in zip(F1_list, label_weights)])
    print('\t', "{:10.3f}".format(precision), \
          "{:10.3f}".format(recall), "{:10.3f}".format(F1))

    # Print overall average accuracy
    print(f'\nAverage Accuracy Over All Folds: {average_accuracy:.3f}')

    

# Function to compute precision, recall and F1 for each label
#  and for any number of labels
# Input: list of gold labels, list of predicted labels (in same order)
# Output: returns lists of precision, recall and F1 for each label
#      (for computing averages across folds and labels)
def eval_measures(gold, predicted, labels):
    
    # these lists have values for each label 
    recall_list = []
    precision_list = []
    F1_list = []

    for lab in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:  TP += 1
            if val == lab and predicted[i] != lab:  FN += 1
            if val != lab and predicted[i] == lab:  FP += 1
            if val != lab and predicted[i] != lab:  TN += 1
        # use these to compute recall, precision, F1
        # for small numbers, guard against dividing by zero in computing measures
        if (TP == 0) or (FP == 0) or (FN == 0):
          recall_list.append (0)
          precision_list.append (0)
          F1_list.append(0)
        else:
          recall = TP / (TP + FP)
          precision = TP / (TP + FN)
          recall_list.append(recall)
          precision_list.append(precision)
          F1_list.append( 2 * (recall * precision) / (recall + precision))

    # the evaluation measures in a table with one row per label
    return (precision_list, recall_list, F1_list)
    
def calculate_vocabulary_density(docs):
    all_tokens = [token for doc, _ in docs for token in doc]
    unique_tokens = set(all_tokens)
    total_tokens = len(all_tokens)
    return len(unique_tokens) / total_tokens if total_tokens > 0 else 0

def processkaggle(dirPath, limitStr):
    # Convert the limit argument from a string to an int
    limit = int(limitStr)
    
    os.chdir(dirPath)
    
    # Read training data
    f = open('./train.tsv', 'r')
    phrasedata = []
    for line in f:
        # Ignore the first line starting with "Phrase" and read all lines
        if not line.startswith('Phrase'):
            line = line.strip()  # Remove trailing whitespace
            # Keep only the phrase text and sentiment label
            phrasedata.append(line.split('\t')[2:4])

    # Randomly shuffle and take a subset of the data
    random.shuffle(phrasedata)
    phraselist = phrasedata[:limit]

    print('Read', len(phrasedata), 'phrases, using', len(phraselist), 'random phrases')

    # Create a list of phrase documents as (list of words, label)
    phrasedocs = []
    for phrase in phraselist:
        tokens = nltk.word_tokenize(phrase[0])  # Tokenize the phrase
        phrasedocs.append((tokens, int(phrase[1])))  # Pair tokens with label

    # Define stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    negation_words = {
        "not", "no", "never", "none", "nothing", "nowhere", "neither", "nor",
        "noone", "rather", "hardly", "scarcely", "rarely","seldom"
         }
    additional_stopwords = {
        "'s", "'re", "'ve", "'d", "maybe", "would", "could", "should", 
        "might", "must", "etc", "ok", "okay", "oh", "uh", "um","also","thing","lot"
    }
    all_stop_words = (stop_words - negation_words).union(additional_stopwords)  # Exclude negation words
    punctuation_pattern = re.compile(r'[\s!"#$%&,\-./:;?@^_`\']{1,}')  # Use punctuation from the string module

    # Filter tokens: lowercase, remove stopwords and punctuation (keep negation words)
    docs = []
    for phrase in phrasedocs:
        filtered_tokens = [
         word.lower() for word in phrase[0]
         if word.lower() not in all_stop_words and not punctuation_pattern.match(word)
         ]
        docs.append((filtered_tokens, phrase[1]))

     # Calculate vocabulary density
    Vocabulary_density = calculate_vocabulary_density(docs)
 
    print(f"Vocabulary Density : {Vocabulary_density:.4f}")
   
    # Continue as usual to compute features and classify
    all_words_list = [word for (sent, cat) in docs for word in sent]
    all_words = nltk.FreqDist(all_words_list)
    print(f'Total unique words : {len(all_words)}')

    # Get the 1500 most frequently appearing keywords in the corpus
    word_items = all_words.most_common(1500)
    word_features = [word for (word, count) in word_items]

    # Extract all bigrams from the corpus
    all_bigrams_list = [bigram for (sent, cat) in docs for bigram in bigrams(sent)]
    all_bigrams = nltk.FreqDist(all_bigrams_list)
    #print(f'Total unique bigrams: {len(all_bigrams)}')

    # Get the 1500 most frequent bigrams
    bigram_items = all_bigrams.most_common(1500)
    bigram_features_list = [bigram for (bigram, count) in bigram_items]
    

    # Create feature sets
    
  # featuresets = [(unigram_features(d, word_features), c) for (d, c) in docs]
  # featuresets = [(unigram_and_negation_features(d, word_features), c) for (d, c) in docs]
  # featuresets = [(pos_tag_features(d, word_features), c) for (d, c) in docs]
  # featuresets = [(bigram_features(d, bigram_features_list), c) for (d, c) in docs]
  # featuresets = [(subjectivity_features(d, positivelist, neutrallist, negativelist), c) for (d, c) in docs]
  # featuresets = [(liwc_features(d, poslist, neglist), c) for (d, c) in docs]
    featuresets = [(combined_features(d, word_features, positivelist, neutrallist, negativelist, poslist, neglist), c) for (d, c) in docs]
  
    
    # Train classifier and show performance in cross-validation
    label_list = [c for (d, c) in docs]
    labels = list(set(label_list))  # Get unique labels
    num_folds = 5
    classifier_type = "Naive Bayes"
    print("Use Classifier: ", classifier_type)
    cross_validation_PRF(num_folds, featuresets, labels)

"""
commandline interface takes a directory name with kaggle subdirectory for train.tsv
   and a limit to the number of kaggle phrases to use
It then processes the files and trains a kaggle movie review sentiment classifier.

"""
if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print ('usage: classifyKaggle.py <corpus-dir> <limit>')
        sys.exit(0)
    processkaggle(sys.argv[1], sys.argv[2])
