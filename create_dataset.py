from dask import dataframe as dd
import tweepy
from twitter_creds import API_KEY,API_SECRET,ACCESS_TOKEN,ACCESS_TOKEN_SECRET
import tweepy
import traceback
import time

#Read the dataset - Don't use pandas, use dask instead
dataset = dd.read_csv('dataverse_files/tweets_stance_sentiment_1outof4.csv')
INPUT_COLUMN_NAME = dataset.columns[0]

#Parse the dataset and transform it into suitable format
data = dataset[INPUT_COLUMN_NAME].str.split('~',expand=True,n=3)
data = data.rename(columns={0:'TweetID', 1:'UserID', 2:'TweetSentiment', 3:'PoliticalStance'})

#Authenticate twitter API
auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

#Collect Tweets
counter = 15
tweets = {}
for row in data.itertuples():
    if counter == 0:
        break
    else:
        counter-=1
    tweet_id = row[1]
    try:
        result = api.get_status(tweet_id)
        tweets[tweet_id] = result.text  
        time.sleep(1)
    except Exception as e:
        print(e)
        continue

print(tweets)