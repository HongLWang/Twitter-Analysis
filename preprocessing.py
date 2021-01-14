#coding=utf-8
# @Time : 21-1-9下午11:26 
# @Author : Honglian WANG


import pandas as pd
import re
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import pickle, nltk
from textblob import TextBlob
from collections import Counter
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
from nltk.corpus import stopwords
from time import time
import texthero as hero


with open('Emoji_Dict.p', 'rb') as fp:
    Emoji_Dict = pickle.load(fp)
Emoji_Dict = {v: k for k, v in Emoji_Dict.items()}


# Converting emojis to words
def convert_emojis(text):
    for emot in UNICODE_EMO:
        text = text.replace(emot, "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()))
        return text
# Converting emoticons to words
def convert_emoticons(text):
    for emot in EMOTICONS:
        text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)
        return text

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def remove_comma(text):
    return " ".join([word for word in str(text).split() if word not in ['“','”', '’','，','。']])

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def html(text):
    return BeautifulSoup(text, "lxml").text

def removeAt(text):
    return " ".join([word for word in str(text).split() if not "@" in word])

lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV} # Pos tag, used Noun, Verb, Adjective and Adverb

def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

# lemmatize doesn't work on transfering vaccines to vaccine.
def extra_lemmatize(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def pipline_processing(df, STOPWORDS=None):


    # order of this operation matters
    s = hero.remove_urls(df)
    s = hero.remove_html_tags(s)
    s = hero.lowercase(s)
    s = s.apply(removeAt)
    s =  hero.remove_punctuation(s)
    s = hero.preprocessing.remove_digits(s, only_blocks=False)
    s = s.apply(convert_emojis)
    s = s.apply(convert_emoticons)
    s = s.apply(lemmatize_words)
    s = hero.remove_stopwords(s, stopwords=STOPWORDS)
    s = hero.remove_whitespace(s)
    s = s.apply(lemmatize_words)

    # s = hero.tokenize(s)

    return s




if __name__ == '__main__':
    # st = time()
    # fp = '/usr/DiskRo/Aris_meeting_data/clean2.csv'
    # df = pd.read_csv(fp, nrows=1400)
    # df = df['tweet']
    # df_processed = pipline_processing(df)

    filepath = '/home/morris/Dropbox/Honglian/Aris_meeting/CodeDemo/tweets1127_1130.csv'

    dataframe = pd.read_csv(filepath)
    dataframe = dataframe[['conversation_id', 'tweet']].dropna(axis=0, how='any', inplace=False)

    freq = dataframe.loc[:, 'conversation_id'].value_counts()
    high_freq = freq[freq > 50]
    dataframe = dataframe[dataframe['conversation_id'].isin(high_freq.index)]
    print ('num of instance to process = ', len(dataframe))


    tweets = dataframe['tweet']
    print (tweets[:10])

    st = time()
    pd = pipline_processing(tweets)
    print(pd[:10])
    print ('time spent = ', time()-st)