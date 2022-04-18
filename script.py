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
    '0x1F602', '0x1F60D', '0x1F62D', '0x1F618', '0x1F60A', '0x1F601', '0x1F629', '0x1F64F',
    '0x1F60F', '0x1F609', '0x1F64C', '0x1F648', '0x1F604', '0x1F612', '0x1F603', '0x1F614',
    '0x1F631', '0x1F61C', '0x1F633', '0x1F621', '0x1F60E', '0x1F622', '0x1F60B', '0x1F64A',
    '0x1F634', '0x1F60C', '0x1F61E', '0x1F606', '0x1F61D', '0x1F62A', '0x1F62B', '0x1F605',
    '0x1F600', '0x1F61A', '0x1F63B', '0x1F625', '0x1F615', '0x1F624', '0x1F608', '0x1F630',
    '0x1F611', '0x1F639', '0x1F620', '0x1F613', '0x1F623', '0x1F610', '0x1F628', '0x1F616', 
    '0x1F637', '0x1F64B', '0x1F61B', '0x1F62C', '0x1F619', '0x1F646', '0x1F645', '0x1F649',
    '0x1F607', '0x1F63F', '0x1F632', '0x1F636', '0x1F635', '0x1F638', '0x1F627', '0x1F62E',
    '0x1F63D', '0x1F640', '0x1F647', '0x1F61F', '0x1F62F', '0x1F626', '0x1F64D', '0x1F63A',
    '0x1F63E', '0x1F63C', '0x1F64E', '0x1F617'
]
dbConnection = sqlite3.connect('datasets/tweets-final.db')
cursor = dbConnection.execute('CREATE TABLE TWEETS(TWEET_ID VARCHAR2(100), USER_ID VARCHAR2(100), TWEET VARCHAR(400),EMOJI VARCHAR(100),TWEET_SENTIMENT VARCHAR(10))')
dbNames = ['datasets/tweets-1.db','datasets/tweets-2.db']
for dbName in dbNames:
    con = sqlite3.connect(dbName)
    cursor_obj = con.cursor()
    cursor_obj.execute('SELECT * FROM TWEETS')
    rows = cursor_obj.fetchall()
    count = 0
    for row in rows:
        emoji_list = ['0x{:X}'.format(ord(c)) for c in row[2] if c in emoji.UNICODE_EMOJI['en']]
        emoji_list = [emo for emo in emoji_list if emo in target_emoji_list]
        if len(emoji_list) >= 1: 
            emoji_list_str = '|'.join(emoji_list)
            cursor.execute("INSERT INTO TWEETS(TWEET_ID, USER_ID, TWEET,EMOJI, TWEET_SENTIMENT) VALUES (?,?,?,?,?)",(row[0],row[1],row[2],emoji_list_str,row[3]))
    dbConnection.commit()
print("[INFO] FINISHED SYNCING DATABASES.")

#STEP 1: DATA PRE-PROCESSING

tweets_df = pd.read_sql("SELECT * FROM TWEETS",dbConnection)
tweets_df.dropna()
emoticons_df = pd.read_csv("datasets/emoticonsRanked.csv")
emojis = emoticons_df["Unicode"]
target_emoticons = emojis.tolist()

tweets_df = tweets_df[tweets_df['TWEET'].apply(tweet_en)]
tweets_df.loc[:, 'PROCESSED_TWEET'] = 0
tweets_df.loc[:, 'TARGET_EMOJIS_LIST'] = 0

process_tweets(tweets_df,target_emoticons)
tweets_df.to_csv("chec_this.csv")

#STEP 2: DERIVE AGGREGATED TEXT SENTIMENT FROM THREE MODELS
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
# tweets_df['nltk_postive_score']=tweets_df['PROCESSED_TWEET'].apply(lambda x : sia.polarity_scores(x)['pos'])
# tweets_df['nltk_negative_score']=tweets_df['PROCESSED_TWEET'].apply(lambda x : sia.polarity_scores(x)['neg'])
# tweets_df['nltk_neutral_score']=tweets_df['PROCESSED_TWEET'].apply(lambda x : sia.polarity_scores(x)['neu'])
tweets_df['nltk_aggregated_score']=tweets_df['PROCESSED_TWEET'].apply(lambda x : sia.polarity_scores(x)['compound'])
tweets_df['spacy_aggregated_score']=tweets_df['PROCESSED_TWEET'].apply(lambda x : detect_sentiment_spacy(x))
# tweets_df['hugging_face_score']=tweets_df['PROCESSED_TWEET'].apply(lambda x : sia.polarity_scores(x)['neu'])

# tweets_df['delta']=tweets_df['TARGET_EMOJIS_LIST'].apply(lambda x : detect_emoji_sentiment(x))
print(tweets_df.head(10))


#STEP 3: DERIVE AGGREGATED EMOTICON SENTIMENT FROM EMOTICON SENTIMENT RANKING
tweets_df['emoji_score']=tweets_df['TARGET_EMOJIS_LIST'].apply(lambda x : detect_emoji_sentiment(x))

#STEP 4: DETERMINE THE DELTA
tweets_df['delta_nltk']= abs(tweets_df['nltk_aggregated_score'] - tweets_df['emoji_score'])
tweets_df['delta_spacy']= abs(tweets_df['spacy_aggregated_score'] - tweets_df['emoji_score'])

#STEP 5: PLOT THE DELTA VALUES TO DETERMINE THE THRESHOLD
import matplotlib.pyplot as plt
tweets_df['delta_nltk'].hist()
tweets_df['delta_spacy'].hist()
plt.show()
