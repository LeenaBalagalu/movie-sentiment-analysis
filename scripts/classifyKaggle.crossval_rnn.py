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
from scipy.sparse import vstack
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, Input, LSTM, Dropout
from tensorflow.keras.models import Sequential



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


def cross_validation_PRF(num_folds, featuresets, test_featuresets, labels, classifier_type):
    num_labels = len(labels) 
    subset_size = int(len(featuresets) / num_folds)
    print('Each fold size:', subset_size)

    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform([features for (features, label) in featuresets])
    y = np.array([label for (features, label) in featuresets])

    # Scale Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape input for RNN: (samples, time steps, features)
    if classifier_type == 'Recurrent Neural Network':
        X_rnn = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    # Initialize RNN model
    model = None
    if classifier_type == 'Recurrent Neural Network':
        model = Sequential([
            Input(shape=(1, X.shape[1])),
            Bidirectional(LSTM(64, activation='tanh', return_sequences=True, dropout=0.2)),
            LSTM(64, activation='tanh', return_sequences=False, dropout=0.2),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(len(labels), activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    total_precision_list = [0] * len(labels)
    total_recall_list = [0] * len(labels)
    total_F1_list = [0] * len(labels)
    total_accuracy = 0

    for fold in range(num_folds):
        test_start = fold * subset_size
        test_end = test_start + subset_size

        if classifier_type == 'Recurrent Neural Network':
            # Prepare training and test sets for RNN
            X_test = X_rnn[test_start:test_end]
            y_test = y[test_start:test_end]
            X_train = np.concatenate((X_rnn[:test_start], X_rnn[test_end:]), axis=0)
            y_train = np.concatenate((y[:test_start], y[test_end:]), axis=0)

            # Convert labels to categorical (one-hot encoding)
            y_train_categorical = to_categorical(y_train, num_classes=len(labels))
            y_test_categorical = to_categorical(y_test, num_classes=len(labels))

            # Early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            # Train the RNN model
            model.fit(X_train, y_train_categorical,
                      epochs=50, batch_size=32, validation_split=0.1,
                      callbacks=[early_stopping], verbose=0)

            # Predict probabilities and take the argmax for class predictions
            y_pred_probs = model.predict(X_test)
            y_pred = np.argmax(y_pred_probs, axis=1)
        else:
            raise NotImplementedError(f"Classifier '{classifier_type}' is not supported in this function.")

        # Evaluate
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=labels, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)

        total_accuracy += accuracy

        for i in range(len(labels)):
            total_precision_list[i] += precision[i]
            total_recall_list[i] += recall[i]
            total_F1_list[i] += f1[i]

    # Average metrics across folds
    precision_list = [total / num_folds for total in total_precision_list]
    recall_list = [total / num_folds for total in total_recall_list]
    F1_list = [total / num_folds for total in total_F1_list]
    average_accuracy = total_accuracy / num_folds

    # Print metrics
    print('\nAverage Precision\tRecall\t\tF1 \tPer Label')
    for i, label in enumerate(labels):
        print(f'{label} \t {precision_list[i]:.3f} \t\t {recall_list[i]:.3f} \t\t {F1_list[i]:.3f}')

    print('\nAverage Accuracy:', average_accuracy)

    # print macro average over all labels - treats each label equally
    print('\nMacro Average Precision\tRecall\t\tF1 \tOver All Labels')
    print('\t', "{:10.3f}".format(sum(precision_list) / num_labels), \
          "{:10.3f}".format(sum(recall_list) / num_labels), \
          "{:10.3f}".format(sum(F1_list) / num_labels))



    # for micro averaging, weight the scores for each label by the number of items
    #    this is better for labels with imbalance
    # first intialize a dictionary for label counts and then count them
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
    precision = sum([a * b for a,b in zip(precision_list, label_weights)])
    recall = sum([a * b for a,b in zip(recall_list, label_weights)])
    F1 = sum([a * b for a,b in zip(F1_list, label_weights)])
    print( '\t', "{:10.3f}".format(precision), \
      "{:10.3f}".format(recall), "{:10.3f}".format(F1))
      
     # Print overall average accuracy
    print(f'\nAverage Accuracy Over All Folds: {average_accuracy:.3f}')
    
     # Predict sentiments for test set
    print("\nPredicting sentiments for test set...")
    test_X = vec.transform([features for (_, features) in test_featuresets])

    if classifier_type == 'Recurrent Neural Network':
        test_X_rnn = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])
        test_predictions_probs = model.predict(test_X_rnn)
        test_predictions = np.argmax(test_predictions_probs, axis=1)

    output_file = 'predictions.csv'
    with open(output_file, 'w') as outfile:
        outfile.write('PhraseID,Sentiment\n')
        for i, (phrase_id, _) in enumerate(test_featuresets):
            outfile.write(f"{phrase_id},{test_predictions[i]}\n")

    print(f"Test predictions saved to {output_file}")
    

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
    #Test Set
  
    f_test = open('./test.tsv', 'r')
  
  # loop over lines in the file and use the first limit of them
    test_phrasedata = []
    for line in f_test:
    # ignore the first line starting with Phrase and read all lines
     if (not line.startswith('Phrase')):
      # remove final end of line character
      line = line.strip()
      columns = line.split('\t')  # Split the line by tabs

      # Ensure the line has at least three columns
      if len(columns) >= 3:
                test_phrasedata.append((columns[0], columns[2]))  # Append only the Phrase column


  # Create list of test phrases as (PhraseId, list of words)
    test_phrasedocs = []
    for phrase in test_phrasedata:
        tokens = nltk.word_tokenize(phrase[1])  # Tokenize the Phrase column
        test_phrasedocs.append((int(phrase[0]), tokens))
      
    test_docs = []
  #remove stop words and punctuations
    for phrase in test_phrasedocs:
        filtered_tokens = [word.lower() for word in phrase[1] 
                           if word.lower() not in all_stop_words and not punctuation_pattern.match(word)]
        test_docs.append((phrase[0], filtered_tokens))  # Retain PhraseId with filtered tokens


     # Calculate vocabulary density
    Vocabulary_density = calculate_vocabulary_density(docs)
 
    #print(f"Vocabulary Density : {Vocabulary_density:.4f}")
    
   
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
    test_featuresets = [(phrase_id, combined_features(d, word_features, positivelist, neutrallist, negativelist, poslist, neglist)) for (phrase_id, d) in test_docs]
    
    # Train classifier and show performance in cross-validation
    label_list = [c for (d, c) in docs]
    labels = list(set(label_list))  # Get unique labels
    num_folds = 5
    classifier_type = "Recurrent Neural Network"
    print("Use Classifier: ", classifier_type)
    cross_validation_PRF(num_folds, featuresets,test_featuresets, labels, classifier_type)

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
