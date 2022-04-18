import emoji
import sqlite3
from nltk.tokenize import TweetTokenizer
import pandas as pd
from langdetect import detect
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk import pos_tag
import os
from utils import *

#STEP -1: REMOVE FILES FROM PREVIOUS RUN
import os
if os.path.exists('datasets/tweets-final.db'):
  os.remove('datasets/tweets-final.db')
else:
  print("The file does not exist")


#STEP 0: SYNC TWEETS DATABASES and FILTER OUT TWEETS WITHOUT EMOTICONS
target_emoji_list = [
    '0x1f602', '0x1f60d', '0x1f62d', '0x1f618', '0x1f60a', '0x1f601', '0x1f629', '0x1f64f',
    '0x1f60f', '0x1f609', '0x1f64c', '0x1f648', '0x1f604', '0x1f612', '0x1f603', '0x1f614',
    '0x1f631', '0x1f61c', '0x1f633', '0x1f621', '0x1f60e', '0x1f622', '0x1f60b', '0x1f64a',
    '0x1f634', '0x1f60c', '0x1f61e', '0x1f606', '0x1f61d', '0x1f62a', '0x1f62b', '0x1f605',
    '0x1f600', '0x1f61a', '0x1f63b', '0x1f625', '0x1f615', '0x1f624', '0x1f608', '0x1f630',
    '0x1f611', '0x1f639', '0x1f620', '0x1f613', '0x1f623', '0x1f610', '0x1f628', '0x1f616', 
    '0x1f637', '0x1f64b', '0x1f61b', '0x1f62c', '0x1f619', '0x1f646', '0x1f645', '0x1f649',
    '0x1f607', '0x1f63f', '0x1f632', '0x1f636', '0x1f635', '0x1f638', '0x1f627', '0x1f62e',
    '0x1f63d', '0x1f640', '0x1f647', '0x1f61f', '0x1f62f', '0x1f626', '0x1f64d', '0x1f63a',
    '0x1f63e', '0x1f63c', '0x1f64e', '0x1f617'
]
dbConnection = sqlite3.connect('datasets/tweets-final.db')
cursor = dbConnection.execute('CREATE TABLE TWEETS(TWEET_ID VARCHAR2(100), USER_ID VARCHAR2(100), TWEET VARCHAR(400),EMOJI VARCHAR(100),TWEET_SENTIMENT VARCHAR(10))')
dbNames = ['datasets/tweets-1.db','datasets/tweets-2.db']
for dbName in dbNames:
    con = sqlite3.connect(dbName)
    cursor_obj = con.cursor()
    cursor_obj.execute('SELECT * FROM TWEETS')
    rows = cursor_obj.fetchall()
    for row in rows:
        emoji_list = ['0x{:X}'.format(ord(c)) for c in row[2] if c in emoji.UNICODE_EMOJI['en']]
        emoji_list = [emo.lower() for emo in emoji_list if emo.lower() in target_emoji_list]
        if len(emoji_list) >= 1:
          emoji_list_str = '|'.join(emoji_list)
          cursor.execute("INSERT INTO TWEETS(TWEET_ID, USER_ID, TWEET,EMOJI, TWEET_SENTIMENT) VALUES (?,?,?,?,?)",(row[0],row[1],row[2],emoji_list_str,row[3]))
    dbConnection.commit()
print("[INFO] FINISHED SYNCING DATABASES.")

#STEP 1: DATA PRE-PROCESSING

tweets_df = pd.read_sql("SELECT * FROM TWEETS",dbConnection)
emoticons_df = pd.read_csv("datasets/emoticonsRanked.csv")
emojis = emoticons_df["Unicode"]
target_emoticons = emojis.tolist()

tweets_df = tweets_df[tweets_df['TWEET'].apply(tweet_en)]
tweets_df.loc[:, 'PROCESSED_TWEET'] = 0
tweets_df.loc[:, 'TARGET_EMOJIS_LIST'] = 0

process_tweets(tweets_df,target_emoticons)


#STEP 2: DERIVE AGGREGATED TEXT SENTIMENT FROM THREE MODELS
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
# tweets_df['nltk_postive_score']=tweets_df['PROCESSED_TWEET'].apply(lambda x : sia.polarity_scores(x)['pos'])
# tweets_df['nltk_negative_score']=tweets_df['PROCESSED_TWEET'].apply(lambda x : sia.polarity_scores(x)['neg'])
# tweets_df['nltk_neutral_score']=tweets_df['PROCESSED_TWEET'].apply(lambda x : sia.polarity_scores(x)['neu'])
tweets_df['nltk_aggregated_score']=tweets_df['PROCESSED_TWEET'].apply(lambda x : sia.polarity_scores(x)['compound'])
tweets_df['spacy_aggregated_score']=tweets_df['PROCESSED_TWEET'].apply(lambda x : detect_sentiment_spacy(x))
tweets_df['hugging_face_score']=tweets_df['PROCESSED_TWEET'].apply(lambda x : hugging_sentiment_scores(x))

# tweets_df['delta']=tweets_df['TARGET_EMOJIS_LIST'].apply(lambda x : detect_emoji_sentiment(x))
print(tweets_df.head(10))


#STEP 3: DERIVE AGGREGATED EMOTICON SENTIMENT FROM EMOTICON SENTIMENT RANKING
tweets_df['emoji_score']=tweets_df['TARGET_EMOJIS_LIST'].apply(lambda x : detect_emoji_sentiment(x))

#STEP 4: DETERMINE THE DELTA
tweets_df['delta_nltk']= abs(tweets_df['nltk_aggregated_score'] - tweets_df['emoji_score'])
tweets_df['delta_spacy']= abs(tweets_df['spacy_aggregated_score'] - tweets_df['emoji_score'])
tweets_df['delta_hugging_face']= abs(tweets_df['hugging_face_score'] - tweets_df['emoji_score'])

tweets_df.to_csv("chec_this.csv")

#STEP 5: PLOT THE DELTA VALUES TO DETERMINE THE THRESHOLD
import matplotlib.pyplot as plt
tweets_df[['delta_nltk','delta_spacy','delta_hugging_face']].plot.hist(stacked=True,histtype = 'step', fill = None)
plt.xlabel('Therehold')
plt.ylabel('Frequency')
plt.show()

# PLOT VERSION2
import matplotlib.pyplot as plt
tweets_df[['delta_nltk','delta_spacy','delta_hugging_face']].plot.hist(stacked=True,alpha=0.7)
plt.xlabel('Therehold')
plt.ylabel('Frequency')
plt.show()
