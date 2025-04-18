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
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from scipy.sparse import csr_matrix


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


def cross_validation_PRF(num_folds, featuresets, labels, tfidf_matrix):
    subset_size = len(featuresets) // num_folds
    print('Each fold size:', subset_size)

    # Initialize totals for macro averages and accuracy
    total_precision_list = [0] * len(labels)
    total_recall_list = [0] * len(labels)
    total_F1_list = [0] * len(labels)
    total_accuracy = 0  # Initialize total accuracy

    # Number of unique labels
    num_labels = len(labels)

    # Extract all labels
    all_labels = [label for _, label in featuresets]

    # Convert custom features to numeric vectors using DictVectorizer
    vectorizer_custom = DictVectorizer(sparse=True)
    all_custom_features = [fs[0] for fs in featuresets]
    custom_features_matrix = vectorizer_custom.fit_transform(all_custom_features)

    for fold in range(num_folds):
        # Split indices for cross-validation
        test_indices = list(range(fold * subset_size, (fold + 1) * subset_size))
        train_indices = list(set(range(len(featuresets))) - set(test_indices))

        # Split TF-IDF features
        X_train_tfidf = tfidf_matrix[train_indices]
        X_test_tfidf = tfidf_matrix[test_indices]

        # Split custom features
        X_train_custom = custom_features_matrix[train_indices]
        X_test_custom = custom_features_matrix[test_indices]

        # Combine TF-IDF and custom features
        X_train = hstack([X_train_tfidf, X_train_custom])
        X_test = hstack([X_test_tfidf, X_test_custom])

        # Labels
        y_train = np.array([all_labels[i] for i in train_indices])
        y_test = np.array([all_labels[i] for i in test_indices])

        # Train XGBoost Classifier
        classifier = XGBClassifier(
            n_estimators=50, 
            max_depth=6, 
            learning_rate=0.1,
            booster='gbtree',
            tree_method='hist',            
            eval_metric='logloss',
            class_weight='balanced',            # Necessary to avoid warnings in recent XGBoost versions
            random_state=42
        )
        classifier.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = classifier.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=labels, average=None)

        # Calculate accuracy for this fold
        fold_accuracy = accuracy_score(y_test, y_pred)
        total_accuracy += fold_accuracy

        #print(f'Fold {fold + 1} Accuracy: {fold_accuracy:.3f}')

        # Update totals
        for i in range(len(labels)):
            total_precision_list[i] += precision[i]
            total_recall_list[i] += recall[i]
            total_F1_list[i] += f1[i]

    # Compute averages
    precision_list = [tot / num_folds for tot in total_precision_list]
    recall_list = [tot / num_folds for tot in total_recall_list]
    F1_list = [tot / num_folds for tot in total_F1_list]
    average_accuracy = total_accuracy / num_folds  # Average accuracy

    # Print averages per label
    print('\nAverage Precision\tRecall\t\tF1 \tPer Label')
    for i, lab in enumerate(labels):
        print(f'{lab} \t {precision_list[i]:.3f} \t\t {recall_list[i]:.3f} \t\t {F1_list[i]:.3f}')

    # Print macro average over all labels
    print('\nMacro Average Precision\tRecall\t\tF1 \tOver All Labels')
    print('\t', f"{sum(precision_list) / num_labels:.3f}\t{sum(recall_list) / num_labels:.3f}\t{sum(F1_list) / num_labels:.3f}")



    # Micro averaging
    label_counts = {lab: 0 for lab in labels}
    for (_, lab) in featuresets:
        label_counts[lab] += 1
    num_docs = len(featuresets)
    label_weights = [(label_counts[lab] / num_docs) for lab in labels]

    print('\nMicro Average Precision\tRecall\t\tF1 \tOver All Labels')
    precision = sum([a * b for a, b in zip(precision_list, label_weights)])
    recall = sum([a * b for a, b in zip(recall_list, label_weights)])
    F1 = sum([a * b for a, b in zip(F1_list, label_weights)])
    print('\t', f"{precision:.3f}\t{recall:.3f}\t{F1:.3f}")
    
       # Print overall average accuracy
    print(f'\nOverall Average Accuracy Across All Folds: {average_accuracy:.3f}')

    

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
    #print(f"Vocabulary Density : {calculate_vocabulary_density(docs):.4f}")
   
    # Continue as usual to compute features and classify
    all_words_list = [word for (sent, cat) in docs for word in sent]
    all_words = nltk.FreqDist(all_words_list)
    print(f'Total unique words : {len(all_words)}')

    # Get the 1500 most frequently appearing keywords in the corpus
    word_items = all_words.most_common(500)
    word_features = [word for (word, count) in word_items]

    # Create raw text and labels for TF-IDF
    raw_text = [' '.join(tokens) for tokens, _ in docs]

    # Compute TF-IDF matrix
    vectorizer_tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer_tfidf.fit_transform(raw_text)
    

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
    classifier_type = "XGBoost"
    print("Use Classifier: ", classifier_type)
    cross_validation_PRF(num_folds, featuresets, labels, tfidf_matrix)

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
