from enum import auto
import itertools
from collections import Counter
from math import isnan, nan

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
nltk.download('punkt')
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
#from yellowbrick.text import FreqDistVisualizer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             hamming_loss, precision_score, recall_score)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

wordnet_lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stopwords = set(stopwords.words('english'))

# pre-processing input data
def preprocess_text(text):
    text = text.replace("\n", " ")
    tokens = nltk.tokenize.word_tokenize(text.lower()) # split string into words (tokens)
    tokens = [t for t in tokens if t.isalpha()] # keep strings with only alphabets
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [stemmer.stem(t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    cleanedText = " ".join(tokens)
    return cleanedText

def vectorise_features(cleaned_x_train, cleaned_x_test):
    print('Vectorization')
    vectorizer = TfidfVectorizer(analyzer='word', max_features=1000)
    vectorised_x_train = vectorizer.fit_transform(cleaned_x_train)
    vectorised_x_test = vectorizer.transform(cleaned_x_test)

    #show_vectorised_data(vectorizer, vectorised_x_train)

    return vectorised_x_train, vectorised_x_test

ModelsPerformance = {}
def metricsReport(modelName, test_labels, predictions):
    accuracy = accuracy_score(test_labels, predictions)

    macro_precision = precision_score(test_labels, predictions, average='macro')
    macro_recall = recall_score(test_labels, predictions, average='macro')
    macro_f1 = f1_score(test_labels, predictions, average='macro')

    micro_precision = precision_score(test_labels, predictions, average='micro')
    micro_recall = recall_score(test_labels, predictions, average='micro')
    micro_f1 = f1_score(test_labels, predictions, average='micro')
    hamLoss = hamming_loss(test_labels, predictions)
    print("------" + modelName + " Model Metrics-----")
    print("Accuracy: {:.4f}\nHamming Loss: {:.4f}\nPrecision:\n  - Macro: {:.4f}\n  - Micro: {:.4f}\nRecall:\n  - Macro: {:.4f}\n  - Micro: {:.4f}\nF1-measure:\n  - Macro: {:.4f}\n  - Micro: {:.4f}"\
          .format(accuracy, hamLoss, macro_precision, micro_precision, macro_recall, micro_recall, macro_f1, micro_f1))
    ModelsPerformance[modelName] = micro_f1

def build_model_xgb(x_train, y_train, x_test, y_test):
    print(y_train.shape)
    xgb_classifier = XGBClassifier(learning_rate=0.5, verbosity=2, random_state=4)
    xgb_classifier.fit(x_train, y_train)
    prediction = xgb_classifier.predict(x_test)
    metricsReport("XGB:", y_test, prediction)
    # Accuracy: 0.7908
    # Hamming Loss: 0.2092

def build_model_svc(x_train, y_train, x_test, y_test):
    svc_classifier = SVC(gamma=0.001)
    svc_classifier.fit(x_train, y_train)

    prediction = svc_classifier.predict(x_test)
    metricsReport("SVC Sq. Hinge Loss", y_test, prediction)
    # Accuracy: 0.0667
    # Hamming Loss: 0.9333

def build_model_rf(x_train, y_train, x_test, y_test):
    print(y_train.shape)
    rf_classifier = RandomForestClassifier(n_jobs=-1)
    rf_classifier.fit(x_train, y_train)
    prediction = rf_classifier.predict(x_test)
    metricsReport("Random Forest", y_test, prediction)
    # Accuracy: 0.8400
    # Hamming Loss: 0.1600

def print_performance():
    print("  Model Name " + " "*10 + "| Micro-F1 Score")
    print("-------------------------------------------")
    for key, value in ModelsPerformance.items():
        print("  " + key, " "*(20-len(key)) + "|", value)
        print("-------------------------------------------")

def open_file(file_name):
    # Load data
    df = pd.read_csv(file_name)
    print('Number of records:', len(df))
    print('Sample data:')
    print(df.head())

    df.dropna(axis=0, inplace=True)

    return df

def main():
    file_name = 'wiseworking4.csv'
    df = open_file(file_name)

    class_names = df['Category'].unique()
    features = df['Description'] 
    target = df['Category']
        
    # print('Cleaning data')
    cleaned_features =  features.apply(preprocess_text)

    print('Dirty data')
    print(features[2])
    print('Clean data:')
    print(cleaned_features[2])

    #train test split
    x_train, x_test, y_train, y_test = train_test_split(cleaned_features, target, random_state=0, test_size=0.25, shuffle=True)
    print('y1:', y_train.shape)

    # Convert label to binary

    lb = LabelEncoder()
    y_train_vec = lb.fit_transform(y_train)
    y_test_vec = lb.transform(y_test)
    print('y2:', y_train_vec.shape)

    # vectorization
    x_train_vec, x_test_vec = vectorise_features(x_train, x_test)

    #build_model_xgb (x_train_vec, y_train_vec, x_test_vec, y_test_vec)
    #build_model_svc (x_train_vec, y_train_vec, x_test_vec, y_test_vec)
    #build_model_rf (x_train_vec, y_train_vec, x_test_vec, y_test_vec)

if __name__ == "__main__":
    main()

