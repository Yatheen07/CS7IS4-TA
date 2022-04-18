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

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')
emoticons_df = pd.read_csv("datasets/emoticonsRanked.csv")

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
            emojis = "|".join([e for e in emojis])
            df.loc[i, 'PROCESSED_TWEET'] = text
            df.loc[i, 'TARGET_EMOJIS_LIST'] = emojis
        else:
            df.drop(i)

def detect_sentiment_spacy(sentence):
    doc = nlp(sentence)
    return doc._.blob.polarity

def detect_emoji_sentiment(value):
    emojis = value.split("|")
    result = 0
    for emo in emojis:
        if emo in target_emoji_list:
            sentiment_score = list(emoticons_df[emoticons_df["Unicode"] == emo.lower()]["Sentiment score\n[-1...+1]"])[0]
            result += sentiment_score
    return result/len(emojis)