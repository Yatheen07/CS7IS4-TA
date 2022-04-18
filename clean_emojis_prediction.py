import re
import sqlite3
import string

import emoji
import pandas as pd
from nltk import TweetTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax

task = 'sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)


def vader_sentiment_scores(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)

    # print("Overall sentiment dictionary is : ", sentiment_dict)
    # print("sentence was rated as ", sentiment_dict['neg'] * 100, "% Negative")
    # print("sentence was rated as ", sentiment_dict['neu'] * 100, "% Neutral")
    # print("sentence was rated as ", sentiment_dict['pos'] * 100, "% Positive")
    #
    # print("Sentence Overall Rated As", end=" ")
    #
    # # decide sentiment as positive, negative and neutral
    # if sentiment_dict['compound'] >= 0.05:
    #     print("Positive")
    #
    # elif sentiment_dict['compound'] <= - 0.05:
    #     print("Negative")
    #
    # else:
    #     print("Neutral")

    return sentiment_dict['compound']


def spacy_sentiment_scores(sentence):
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob')
    doc = nlp(sentence)
    return doc._.blob.polarity


def hugging_sentiment_scores(sentence):
    encoded_input = tokenizer(sentence, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = list(softmax(scores))
    max_value = max(scores)
    return -max_value if scores.index(max_value) == 2 else max_value


def tweet_tokenization(sentence):
    tweet_tokenizer = TweetTokenizer()
    tokenized_text = tweet_tokenizer.tokenize(sentence.lower())
    punctuation_list = list(string.punctuation)
    tokenized_text = [text for text in tokenized_text if text not in punctuation_list]
    return tokenized_text


dbConnection = sqlite3.connect('database/tweets-final.db')
cursor = dbConnection.cursor()
# cursor.execute(f"""ALTER TABLE TWEETS ADD COLUMN NLTK_SENTIMENT 'float'""")
# cursor.execute(f"""ALTER TABLE TWEETS ADD COLUMN SPACY_SENTIMENT 'float'""")
# cursor.execute(f"""ALTER TABLE TWEETS ADD COLUMN HUGGING_SENTIMENT 'float'""")
cursor.execute('SELECT TWEET_ID, TWEET FROM TWEETS')
rows = cursor.fetchall()
for row in rows:
    tweet_id = row[0]
    content = row[1]
    # remove emojis
    content_p = emoji.replace_emoji(content, replace='')
    # remove mention & url
    content_p = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", content_p)
    content_p = " ".join(content_p.split())
    vader_scores = vader_sentiment_scores(content_p)
    spacy_scores = spacy_sentiment_scores(content_p)
    hugging_scores = hugging_sentiment_scores(content_p)
    print(type(hugging_scores))
    cursor.execute('UPDATE TWEETS SET NLTK_SENTIMENT = ?, SPACY_SENTIMENT = ?, HUGGING_SENTIMENT = ? WHERE TWEET_ID = ?', (vader_scores, spacy_scores, hugging_scores.item(), tweet_id))
    dbConnection.commit()

tableData = pd.read_sql('SELECT TWEET FROM TWEETS', dbConnection)

cursor.close()
dbConnection.close()
