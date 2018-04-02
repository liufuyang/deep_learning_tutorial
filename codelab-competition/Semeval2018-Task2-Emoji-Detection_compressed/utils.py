import pandas as pd
import re

import preprocessor as p
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from string import punctuation
from nltk.stem.snowball import EnglishStemmer

p.set_options(p.OPT.URL)
stemmer = EnglishStemmer()

def text_feature_clean_fun(s):
    s = p.clean(s)
    
    s = re.sub("[^a-zA-Z]+", ' ', s)   # remove numbers
    s = s.lower()
    
    s = ' '.join([stemmer.stem(word) for word in s.split() if word not in (stopwords.words('english'))])
    
    return s


def feature_transform(raw_feature_str_list, vectorizer, tfidf_transformer):
    X = [text_feature_clean_fun(row)for row in raw_feature_str_list]
    
    X = vectorizer.transform(X)
    X = tfidf_transformer.transform(X)
    
    return X
    

def load_data():
    with open('train_us_semeval18_tweets.json.labels') as f:
        df_train_label = f.read().splitlines()

    # !sed 's/$/ foofuyangty/' train_us_semeval18_tweets.json.text > new.txt
    with open('new.txt') as f:
        df_train = f.read()
        print('count {}'.format(df_train.count('foofuyangty')))
        print('count \\n {}'.format(df_train.count('\n')))
        df_train = df_train.split('  foofuyangty\n')
        df_train=df_train[:-1]
        print(len(df_train))
        
    with open('us_trial.labels') as f:
        df_test_label = f.read().splitlines()
    with open('us_trial.text') as f:
        df_test = f.read().splitlines()
        
    df_train = pd.DataFrame({
        'text': df_train,
        'label': df_train_label})

    df_test = pd.DataFrame({
        'text': df_test,
        'label': df_test_label})
        
    df_train['label'] =  df_train['label'].apply(int)
    df_test['label'] = df_test['label'].apply(int)
        
    return df_train, df_test