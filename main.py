import pandas as pd
import nltk
import numpy as np
import re
from nltk.stem import wordnet # to perform lemmitization
from sklearn.feature_extraction.text import CountVectorizer # to perform bow
from sklearn.feature_extraction.text import TfidfVectorizer # to perform tfidf
from nltk import pos_tag # for parts of speech
from sklearn.metrics import pairwise_distances # to perform cosine similarity
from nltk import word_tokenize # to create tokens
from nltk.corpus import stopwords # for stop words

# function that perform text normalization 
def text_normalization(text):
    text = str(text).lower() # text to lower case
    spl_char_text = re.sub(r'[^ a-z]', '', text) # removing special characters
    tokens = nltk.word_tokenize(spl_char_text) # word tokenizing
    lema = wordnet.WordNetLemmatizer() # initializing lemmatization
    tags_list = pos_tag(tokens, tagset=None) # parts of speech
    lema_words = [] # empty list
    for token, pos_token in tags_list:
        if pos_token.startswith('V'): # verb
            pos_val = 'v'
        elif pos_token.startswith('J'): # adjective
            pos_val = 'a'
        elif pos_token.startswith('R'): # adverb
            pos_val = 'r'
        else:
            pos_val = 'n' # noun
        lema_token = lema.lemmatize(token,pos_val) # performing lemmatization
        lema_words.append(lema_token) # appending the lemmatized token into a list
    return " ".join(lema_words) # return the lemmatized as a sentence

# define a function that returns response to query
def chat_tfidf(text):
    lemma = text_normalization(text) # calling the function to perform text normalization
    tfidf = TfidfVectorizer() # initializing tf-idf
    x_tfidf = tfidf.fit_transform(df['lemmatized_text']).toarray() # transforming the data into array
    df_tfidf = pd.DataFrame(x_tfidf, columns=tfidf.get_feature_names()) # return all the unique word from data
    tf = tfidf.transform([lemma]).toarray() # applying tf-idf
    cos = 1-pairwise_distances(df_tfidf, tf, metric='cosine') # applying cosine similarity
    index_value = cos.argmax() # getting index value
    return df['Text Response'].loc[index_value]

if __name__ == "__main__":
    df = pd.read_excel('dialog_talk_agent.xlsx')
    # print(df.shape[0])
    df.ffill(axis=0, inplace=True) # fills the null value with the previos value
    # print(text_normalization('telling you some stuff about me'))
    df['lemmatized_text']=df['Context'].apply(text_normalization) # apply the function to the dataset
    # print(df.head(5))

     # bag of words
    cv = CountVectorizer() # initializing the count vectorizer
    X = cv.fit_transform(df['lemmatized_text']).toarray()

     # return all the unique word from data
    features = cv.get_feature_names()
    df_bow = pd.DataFrame(X, columns=features)
    # print(df_bow.head(5))

    while(True):
        question = input()
        print(chat_tfidf(question))