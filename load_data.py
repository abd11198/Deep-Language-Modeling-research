import nltk
from nltk.corpus import shakespeare, brown
from nltk.tokenize import word_tokenize, TweetTokenizer
import emoji
import string
import re
import random

nltk.download('punkt')
"""
This class is used to generate the corpuses to be used during training of our two models.
It produces 3 json files for each of the three styles of text (Formal, Informal, and Shakesperean)
with 20,000-30,000 text samples being saved for each respective style. 
NLTK corpora are used to generate Shakesperean and Formal language samples, whereas the Informal language samples 
are scraped from twitter and formalized courtesy of https://github.com/marsan-ma/chat_corpus/
"""

#Generates samples of Shakespearean english
def gen_shakespearean():
    nltk.download('shakespeare')
    samples = []
    for id in shakespeare.fileids():
        play = shakespeare.xml(id)
        for sents in play.findall('.//SPEECH'):
            for line in sents.findall('.//LINE'):
                if line.text is not None:
                    samples.append(line.text)

    return samples

#Generates samples of Formal english
def gen_formal():
    nltk.download('brown')
    samples = []
    sentences = []
    for category in ['learned', 'news', 'government','editorial','reviews']: #formal writing categories in brown corpus
        sentences.extend(brown.sents(categories=category))

    for sent in sentences:
        if len(sent)>2:
            sent = ' '.join(sent)
            samples.append(sent)

    return samples

#Generates samples of Informal english
def gen_informal():
    path= '../twitter samples/chat.txt'
    with open(path,"r", encoding='utf-8') as informal_txt:
        twts = informal_txt.readlines()
    
    samples = random.sample(twts,int(3e4))#take 30k random samples of original 700k

    return samples 

#Preprocessing methods:

def clean_tweet(tokenized_tweet):
  clean_tweet = []
  for token in tokenized_tweet:
    if re.match(r'@',token) and len(token)>1:
      token = '@USER'
    elif re.match(r'http\S+|https\S+|www.\S+',token):
      token = 'URL'
    clean_tweet.append(emoji.demojize(token))

  return(clean_tweet)

def preprocess_samples(samples, isTweet, style_tag):
    processed_samples = []
    for sample_text in samples:
        if isTweet:
            tokenizer = TweetTokenizer(strip_handles=False)
            tokens = tokenizer.tokenize(sample_text)
            tokens = clean_tweet(tokens)
        else:
            tokens = word_tokenize(sample_text)

        tokens = [token for token in tokens if token not in string.punctuation]
        tokens.insert(0,style_tag)
        processed_samples.append(' '.join(tokens))     
    return processed_samples

def generate_data():
    shakespeare_data = gen_shakespearean()
    formal_data = gen_formal()
    informal_data = gen_informal()

    clean_sd = preprocess_samples(shakespeare_data,False,'[SHK]')
    clean_fd = preprocess_samples(formal_data,False,'[FRM]')
    clean_id = preprocess_samples(informal_data,True,'[INF]')

    #23888 shakespearean samples | 19458 formal samples | 30000 informal samples
    #73346 samples total

    with open('data/shakespeare_text.txt','w', encoding='utf-8') as shk_txt:
        for samp in clean_sd:
            shk_txt.write(samp)
            shk_txt.write('\n')
    with open('data/formal_text.txt','w', encoding='utf-8') as frm_txt:
        for samp in clean_fd:
            frm_txt.write(samp)
            frm_txt.write('\n')
    with open('data/informal_text.txt','w', encoding='utf-8') as inf_txt:
        for samp in clean_id:
            inf_txt.write(samp)
            inf_txt.write('\n')

generate_data()