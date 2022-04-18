from nltk.tokenize import TweetTokenizer
from numpy import unicode_
import pandas as pd
from langdetect import detect
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk import pos_tag
import emoji
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
tokenizer.save_pretrained(MODEL)


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')
emoticons_df = pd.read_csv("datasets/emoticonsRanked.csv")

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

def tweet_en(text):
    try:
        return detect(text) == 'en'
    except:
        return False

def wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    else:
        # default
        return wordnet.NOUN


def words_lemmatization(pos_tagged):
    words = []
    for word, tag in pos_tagged:
        lem = lemmatizer.lemmatize(word, wordnet_tag(tag))
        words.append(lem)
    return words


# TODO:a list of emojis with sentiment
def clean_tweet(str,target_emoticons):
    # remove urls
    str = re.sub(r'http\S+', '', str)
    # remove mentions
    str = re.sub(r'@\S+', '', str)
    # remove hashtags
    str = re.sub(r'#\S+', '', str)
    # tokenize
    tweet_tokenizer = TweetTokenizer()
    tokenized_text = tweet_tokenizer.tokenize(str.lower())
    # remove stopwords
    tokenized_text = [w for w in tokenized_text if w.lower() not in stop_words]
    # get pos tags
    pos_tagged = pos_tag(tokenized_text)
    # lemmatization
    tokenized_text = words_lemmatization(pos_tagged)
    # demojize if not in high sentiment list
    demojized_text = ' '.join([emoji.demojize(c) for c in tokenized_text if
                               c not in target_emoticons])
    # remove punctuation
    str = re.sub(r'[^\w\s]', '', demojized_text)
    return str


# TODO: extract emoticons e.g. :), :(
def extract_emojis(s):
    emoji_list = ['0x{:X}'.format(ord(c)) for c in s if c in emoji.UNICODE_EMOJI['en']]
    # emoji_list=[c.lower() for c in emoji_list if c.lower() in target_emoticons]
    # smilies_list = [c for c in s if c in smilies]
    return emoji_list

def process_tweets(df,target_emoticons):
    for i, row in df.iterrows():
        emojis = extract_emojis(row['TWEET'])
        emoji_list = [emo.lower() for emo in emojis if emo.lower() in target_emoticons]
        if len(emoji_list) > 0:
            text = clean_tweet(row['TWEET'],target_emoticons)
            emojis_list = "|".join([e for e in emoji_list])
            df.loc[i, 'PROCESSED_TWEET'] = text
            df.loc[i, 'TARGET_EMOJIS_LIST'] = emojis_list
        else:
            df.drop(i)

def detect_sentiment_spacy(sentence):
    doc = nlp(sentence)
    return doc._.blob.polarity

def hugging_sentiment_scores(sentence):
    encoded_input = tokenizer(sentence, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = list(softmax(scores))
    max_value = max(scores)
    return -max_value if scores.index(max_value) == 2 else max_value



def detect_emoji_sentiment(value):
    emojis = value.split("|")
    result = 0
    for emo in emojis:
        if emo in target_emoji_list:
            sentiment_score = list(emoticons_df[emoticons_df["Unicode"] == emo.lower()]["Sentiment score\r\n[-1...+1]"])[0]
            result += sentiment_score
    return result/len(emojis)