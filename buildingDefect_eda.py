# https://towardsdatascience.com/no-labels-no-problem-30024984681d
# https://machinelearningmastery.com/expectation-maximization-em-algorithm/

from math import isnan, nan
from os import replace
from warnings import catch_warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
# import texthero as hero
# from texthero import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
import itertools
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
#from yellowbrick.text import FreqDistVisualizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, hamming_loss
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
#from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from skmultilearn.problem_transform import LabelPowerset
import spacy
import re
import time, datetime

def plot_data(keys, values, title):
    plt.figure(figsize=(8,4))
    plt.bar(keys, values)
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.3)
    plt.title = title
    plt.show()

 
def open_file(file_name):
    # Load data
    df = pd.read_csv(file_name)
    # print('Number of records:', len(df))
    # print('Sample data:')
    # print(df.head())

    # num_of_recs = len(df)

    # df_check_null = df.isna().sum().to_frame().reset_index()
    # df_check_null = df_check_null.rename(columns={'index': 'column', 0:'null_count'})
    # df_check_null['not_null_count'] = num_of_recs - df_check_null['null_count']

    #print(df.loc[df.Category.isna(),['Project','Location','Category']])

    return df

def get_insight_category(df):
    #df_Category = df.groupby('Category')['Category'].nunique()

    print('Number of unique values in each column')
    print(df.nunique(axis=0))
    print('\n')

    df_category = df.Category.value_counts().to_frame().reset_index()
    df_category.columns = ['Category', 'Count']
    print('Number of records per Category')
    print(df_category)
    print('\n')

    plot_data(df_category.Category, df_category.Count, 'Category')

def get_insight_status(df):
    df_status = df.Status.value_counts().to_frame().reset_index()
    df_status.columns = ['Status', 'Count']
    print('Number of records per Category')
    print(df_status)
    print('\n')

    plot_data(df_status.Status, df_status.Count, 'Status')


def open_excel_file(file_name):
    column_names = ['Project', 'Location', 'Date Raised', 'Rectified Date', 'Category',
       'Subcategory', 'Root Cause', 'Cost Attribute', 'Status', 'Description']

    df = pd.DataFrame(columns=column_names)

    # Load data
    excel_file = pd.ExcelFile(file_name)
    sheet_names = excel_file.sheet_names
    print('Sheet names:', sheet_names)

    for sheet in sheet_names:
        df_excel = excel_file.parse(sheet)
        df_sheet = pd.DataFrame(columns=column_names)
        for col in column_names:
            if col in df_excel.columns:
                df_sheet[col] = df_excel[col]
        df = df.append(df_sheet, ignore_index=True)
    return df

def clean_category(df):
    #df['Category'] = df['Category'].fillna(value='No Category')

    df = df[df.Category != 'No Defect/Damage']

    df.loc[df.Category.str.contains(' / '), 'Category'] = df.Category.str.replace(' / ', '/')
    df.loc[df.Category.str.contains('/ '), 'Category'] = df.Category.str.replace('/ ', '/')

    mylist = ['Balustrades','Lifts','Shower Screens']
    pattern = '|'.join(mylist)
    df.loc[df.Category.str.contains(pattern), 'Category'] = df.Category.str.rstrip('s')

    df.loc[df.Category=='Balustrading', 'Category'] = 'Balustrade'

    df.loc[df.Category=='Windows/FaÃ§ade', 'Category'] = 'Windows/Facade'

    print(sorted(df['Category'].unique()))

    return df

def move_categories(df, old_category, new_category):
    df.loc[df.Category == old_category, 'Category'] = new_category
    return df

def make_data_for_prediction(df):
    categories_for_prediction = ['No Category', 'Misc', 'Defect']
    df = df[df.Category.isin(categories_for_prediction)]
    df = df[['Category','Description']]
    return df

def clean_category_for_model(df):
    unused_categories = ['No Category', 'Misc', 'No Defect/Damaged', 'New Type', 'Defect', 'Inspection Defect', 'Signage', 'Cleaning']
    df = df[~df.Category.isin(unused_categories)]

    # move similar category
    move_categories(df, 'Windows', 'Windows/Facade')
    move_categories(df, 'Facade', 'Windows/Facade')
    move_categories(df, 'Fire', 'Fire Services')
    move_categories(df, 'Fire Pipe', 'Fire Services')
    move_categories(df, 'Tiling', 'Tile/Stone/Caulking')
    move_categories(df, 'Tile/Stone', 'Tile/Stone/Caulking')

    # remove records that it's category has less than 10 entities
    value_counts = df.Category.value_counts()
    to_keep = value_counts[value_counts >= 10].index
    df = df[df.Category.isin(to_keep)]

    return df

def calculate_response_days(df):
    df.dropna(subset=['Rectified Date'], inplace = True)
    df.drop(df.loc[df['Date Raised']=='0000-00-00'].index, inplace=True)

    for index, row in df.iterrows():
        rectified_date = split_convert_date(row['Rectified Date'])
        date_raised = convert_date(row['Date Raised'])
        df.loc[index, 'rectified_date'] = rectified_date
        df.loc[index, 'date_raised'] = date_raised

        date_format = "%m/%d/%Y"
        date_raised_dt = datetime.date.fromtimestamp(time.mktime(time.strptime(date_raised, "%d/%m/%Y")))
        rectified_date_dt = datetime.date.fromtimestamp(time.mktime(time.strptime(rectified_date, "%d/%m/%Y")))
        df.loc[index, 'response_days'] = (rectified_date_dt - date_raised_dt).days

    return df

def extract_location(df):
    import inflect
   
    df[['room_location']] = ''

    # replace mistype
    df['Location'] = df['Location'].str.replace(':evel','Level')

    # Bulk set room location
    df.loc[df.Location.str.contains('ground', case=False), 'room_location'] = 'ground'
    df.loc[df.Location.str.contains('unit g', case=False), 'room_location'] = 'ground'
    df.loc[df.Location.str.contains('common', case=False), 'room_location'] = 'common'
    df.loc[df.Location.str.contains('basement', case=False), 'room_location'] = 'basement'
    df.loc[df.Location.str.contains('TH'), 'room_location'] = 'townhouse'
    df.loc[df.Location.str.contains('townhouse', case=False), 'room_location'] = 'townhouse'
    df.loc[df.Location.str.contains('roof', case=False), 'room_location'] = 'roof'

    # extract level by ordinal numbers
    print_once = True
    for index, row in df.iterrows():
        if row['room_location'] == '':
            x = re.search(r"(level\s?)([1-9][0-9]|0?[1-9]|0)", row['Location'], re.IGNORECASE)
            if x != None:
                if x.group(2) == '0':
                    df.loc[index, 'room_location'] = 'ground'
                else:
                    df.loc[index, 'room_location'] = int(x.group(2))

        if row['room_location'] == '':
            x = re.search(r"(unit |\w*[A-Z])([1-9][0-9][0-9]|0?[1-9][0-9]|0?[1-9])", row['Location'], re.IGNORECASE)
            if x != None:
                if len(x.group(2)) == 3:
                    # example Unit 301 will be on level 3
                    lvl = x.group(2)[0:1]
                    df.loc[index, 'room_location'] = int(lvl)
                else:
                    # example unit 15 will be on ground floor
                    df.loc[index, 'room_location'] = 'ground'

        # detect first, second, etc
        if row['room_location'] == '':  
            p = inflect.engine()
            for i in range(1,10):
                level_ordinal = p.number_to_words(p.ordinal(i))
                if level_ordinal in row['Location'].lower():
                    df.loc[index, 'room_location'] = i

    return df

def clean_text(sentence):          
    #line = re.sub(r'[^ \s\w\.\,]', '', sentence).lower()
    #line = re.sub(r'rn', '. ', line)
    new_sentence = re.sub(r'log|_x000d_|_x000D_|#NAME\?|Ã¢â‚¬Â¦', '', sentence)
    return new_sentence

def clean_description(df):

    df.dropna(subset=['Description'], inplace = True)

    df = df[~df.Description.str.contains('Testing entry|Testing email')]

    # Drop description that has less than 2 words
    # Remove junk words from description
    for index, row in df.iterrows():
        description = clean_text(row['Description'])
        df.loc[index, 'Description'] = description
        if (len(description.split(' ')) < 2): 
            df.loc[index, 'Description'] = ''

    # Drop empty description
    df = df[df.Description != '']

    return df

def augment_text(df):
    # Joinery 1466, Signage 10
    print('augment_text')
    import nltk
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    import nlpaug
    import nlpaug.augmenter.word as naw
    import math

    aug = naw.SynonymAug(aug_src='wordnet',aug_max=10) # maximum 10 words to be changed

    # get only column category and description
    df = df[['Category','Description']]

    # augment only if cases less than 100
    df_category = df.Category.value_counts().to_frame().reset_index()
    df_category.columns = ['Category', 'Count']
    df_aug = df_category[df_category.Count < 1000]

    for entry, row_aug in df_aug.iterrows():
        aug_cat = row_aug['Category']
        print(aug_cat)
        aug_num = 1000 - row_aug['Count'] #total number of text to be augmented
        data = df[df.Category == aug_cat] # collect data per category
        num_rows_per_category = len(data) # number of data per category
        aug_num_per_text = math.ceil(aug_num / num_rows_per_category) # how many augmentation needed
        print(num_rows_per_category, aug_num_per_text)
        for index, row in data.iterrows():
            new_text_list = aug.augment(row['Description'],n=aug_num_per_text)
            if type(new_text_list) is str: 
                new_text_list = [new_text_list]
            for new_text in new_text_list:
                dict = {'Category': aug_cat, 'Description': new_text}
                df = df.append(dict, ignore_index = True)

    return df

def main():
    process_index = 'aug' # merge, report, clean, ready, aug, predict

    excel_file = 'Deakin Requested Defect List (Projects 1-5).xlsx'
    csv_file = 'wiseworking.csv' # merge 3 excel sheets into 
    file_report = 'wiseworking_report.csv' # data for reporting
    file_clean = 'wiseworking_clean.csv' # clean data
    file_ready = 'wiseworking_ready.csv' # ready for modelling
    file_train = 'wiseworking_train.csv' # data augmentation
    file_test = 'wiseworking_test.csv' # data augmentation
    file_predict = 'wiseworking_predict.csv' # after augmentation and only desc and category columns

    if process_index == 'merge':
        df = open_excel_file(excel_file)
        df.to_csv(csv_file, index = False)
        get_insight_category(df)
        get_insight_status(df)

    if process_index == 'report':
        df = open_file(csv_file)
        df = extract_location (df)
        df.to_csv(file_report, index = False)

    if process_index == 'clean':
        df = open_file(file_report)
        df = clean_category(df)
        df = clean_description(df)
        df = calculate_response_days(df)
        df.to_csv(file_clean, index = False)

    if process_index == 'ready':
        df = open_file(file_clean)
        df = clean_category_for_model(df)
        df.to_csv(file_ready, index = False)

    if process_index == 'aug':
        df = open_file(file_ready) 

        # divide data into train and test

        X = df[['Description']]
        y = df[['Category']]

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 1, stratify = df['Category'])

        df_test = pd.DataFrame(columns=['Category', 'Description'])
        df_test['Category'] = y_test
        df_test['Description'] = X_test
        df_test.to_csv(file_test, index = False)

        df_train = pd.DataFrame(columns=['Category', 'Description'])
        df_train['Category'] = y_train
        df_train['Description'] = X_train

        df = augment_text(df_train)
        get_insight_category(df)
        df.to_csv(file_train, index = False)

    if process_index == 'predict':
        df = open_file(file_clean)
        df = make_data_for_prediction (df)
        df.to_csv(file_predict, index = False)

    print('Finish ', process_index)

def test_extract_nouns():
    nlp = spacy.load("en_core_web_sm")
    #doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
    doc = nlp("Tenants advised bathroom and laundry exhaust fans not working Also there is a hollow in the floor near to front door and the floor is bouncy. Tenants Adam              and Laura")          

    for np in doc.noun_chunks:
        print(np.text)

    # for token in doc:
    #     print('token text:', token.text)
    #     #print('token lemma:', token.lemma)
    #     #print('token post:', token.pos)
    #     print('token tag:',token.tag_)
    #     print('token dep:', token.dep_)
    #     #print('token shape:',token.shape_)
    #     #print('is alpha?', token.is_alpha)
    #     #print('is stop?',token.is_stop)
    #     print('\n')


def check_sentence(sentence):
    nlp = spacy.load("en_core_web_sm")

    #sentence = "Tenants advised bathroom and laundry exhaust fans not working Also there is a hollow in the floor near to front door and the floor is bouncy. Tenants Adam              and Laura"       
    #sentence = "#NAME?"
    #sentence = "Oven not working Tenants Adam              and Laura             "
    #sentence = "Hi Justin"
    doc = nlp(sentence)

    has_noun = False
    has_verb = False
    has_adj = False
    is_sentence_ok = False

    for chunk in doc.sents:
        for tok in chunk:
            print(tok, tok.pos_)
            if tok.pos_ == "NOUN" or tok.pos_ == "PROPN":
                has_noun = True
            if tok.pos_ == "ADJ":
                has_adj = True
            if tok.pos_ == "VERB":
                has_verb = True
            if (has_noun and has_adj) or (has_noun and has_verb) or (has_adj and has_verb):
                #print('hi')
                is_sentence_ok = True
                break
        if is_sentence_ok:
            #print('hi2')
            break

    if not is_sentence_ok:
        print(sentence)

    return is_sentence_ok


def test_spacy_graph():
    from spacy import displacy 
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Toilet flushing issue") 
    displacy.serve(doc, style="dep")

def convert_date(date_text):
    date_patterns = ["%d/%m/%Y", "%d-%m-%Y","%Y-%m-%d"]
    target_format = "%d/%m/%Y"
    date_time_string = date_text.split()
    date_string = date_time_string[0]
    
    # Find date pattern
    original_format = ''
    for date_pattern in date_patterns:
        if original_format == '':
            try:
                time.strptime(date_string, date_pattern)
                original_format = date_pattern
            except ValueError:
                pass

    # convert date to certain date format    
    converted_date = date_string
    if original_format == '':
        print('missing format for this date: ', converted_date)
    else:
        converted_date = datetime.datetime.strptime(date_string,original_format).strftime(target_format)
    #print('new date:', converted_date)
    return converted_date

def split_convert_date(date_text):
    date_list = date_text.split(">")
    new_dates = []
    for date_text in date_list:
        new_dates.append(convert_date(date_text.strip()))
    new_dates.sort(key=lambda x: time.mktime(time.strptime(x,'%d/%m/%Y')), reverse=True)
    return new_dates[0]

if __name__ == "__main__":
    main()
