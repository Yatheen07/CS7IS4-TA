import emoji
from nltk.tokenize import TweetTokenizer
import pandas as pd
from langdetect import detect
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk import pos_tag
import os

# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

df = pd.read_csv("datasets/tweets_final_csv.csv")
emoticons_df = pd.read_csv("datasets/emoticonsRanked.csv")
df.dropna()

emojis = emoticons_df["Unicode"]
target_emoticons = emojis.tolist()


def tweet_en(text):
    try:
        return detect(text) == 'en'
    except:
        return False


df = df[df['TWEET'].apply(tweet_en)]


# test = {
#     '1': "🤔🙈 Some text, more 😌 emojis 👨‍👩‍ hello 👩🏾‍🎓 emoji hello 👨‍👩‍👦‍👦 emojis 😊 text analytics🙅🏽🙅🏽",
#     '2': 'TA👍🤔🙈🙈😕', '3': 'emoticon :)'}


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
def tweet_tokenization(str):
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


df.loc[:, 'PROCESSED_TWEET'] = 0
df.loc[:, 'TARGET_EMOJIS_LIST'] = 0


def process_tweets():
    for i, row in df.iterrows():
        emojis = extract_emojis(row['TWEET'])
        emoji_list = [emo.lower() for emo in emojis if emo.lower() in target_emoticons]
        if len(emoji_list) > 0:
            text = tweet_tokenization(row['TWEET'])
            emojis = "|".join([e for e in emojis])
            df.loc[i, 'PROCESSED_TWEET'] = text
            df.loc[i, 'TARGET_EMOJIS_LIST'] = emojis
        else:
            df.drop(i)


process_tweets()

filename = "tweets_processed.csv"
outdir = './datasets'
if not os.path.exists(outdir):
    os.mkdir(outdir)
fullname = os.path.join(outdir, filename)
df.to_csv(fullname, index=False)
print("Saving dataset")
