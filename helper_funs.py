from bs4 import BeautifulSoup as BS
from urllib.request import Request, urlopen
import pickle as pickle
import re
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import string
from datetime import datetime as dt
import pandas as pd
from urllib.error import HTTPError
import os

def get_url_content(url):
    '''
    function that returns BS object from given link
    '''
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urlopen(req)
    bs = BS(html.read(), 'html.parser')
    return(bs)

def get_arabic_from_roman(user_roman):
    
    roman_numerals = { 
    'M': 1000,
    'CM': 900,
    'D': 500,
    'CD': 400,
    'C': 100,
    'XC': 90,
    'L': 50,
    'XL': 40,
    'X': 10,
    'IX': 9,
    'V': 5,
    'IV': 4,
    'I': 1}
    
    user_roman = user_roman.upper()
    resultI = 0

    while user_roman:
        # try first two chars
        if user_roman[:2] in roman_numerals.keys():
            resultI += roman_numerals[user_roman[:2]]
            # cut off first two chars
            user_roman = user_roman[2:]
        # try first char
        elif user_roman[:1] in roman_numerals:
            resultI += roman_numerals[user_roman[:1]]
            # cut off first char
            user_roman = user_roman[1:]
        else:
            print('No roman number')
            return
    print(resultI)

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return(start)

def rfind_nth(haystack, needle, n):
    end = haystack.rfind(needle)
    while end >= 0 and n > 1:
        start = haystack.rfind(needle, 0,end-len(needle))
        n -= 1
    return(start)

def flatten(t):
    return [item for sublist in t for item in sublist]

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

def preprocess_text(text,
                    cleanup = False,
                    stopwords = False,
                    tokenize = False,
                    lemmatize = False,
                    add_stopwords = [],
                    add_chars_to_clean = []
                    ):
    if(cleanup):
        
        # remove punctuation
        text = re.sub(r'[^\w\s]','',text)
        
        # text = text.translate(str.maketrans('', '', string.punctuation)) # works faster but not vectorized over Series
        
        # remove digits
        
        text = re.sub('\d+','', text)
        
        # special characters to spaces
        text = re.sub('\s+',' ',text)
        
        # additional provided characters
        if add_chars_to_clean != []:
            for char in add_chars_to_clean:
                text = re.sub(char,' ',text)
        
        # single character words (most are meaningless)
        text = re.sub(r"\b[a-zA-Z]\b", "", text)
        
        # multiple spaces to single spaces
        text = re.sub("/ +/", " ",text)
        
        # to lower
        text = text.lower()
        
    if(stopwords):
        stop = nltk.corpus.stopwords.words('english')
        
        if add_stopwords != []:
            stop = stop + add_stopwords  
        text = ' '.join([word for word in text.split() if word not in (stop)])
    
    if(tokenize):
        text = nltk.tokenize.word_tokenize(text)
    
    if(lemmatize and tokenize):
        
        text = [get_lemma(word) for word in text]
        
    if(lemmatize and not tokenize):
        print("Warning!\nLLemmatization only allowed together with tokenization. Lemmatization skipped!")
    
    return(text)

def get_top_k_ngrams(data,k = 5, n = 1, freq = False):
    
    dist = nltk.FreqDist(nltk.ngrams(data, n))
    if not freq:
        
        return(dist.most_common(k))
    else:
        big_list = dist.most_common(k)
        final_list = [tuple([lst[i] for i in range(len(lst)-1)]) + tuple([n/len(data) for n in lst[-1:]]) for lst in big_list ]
        return(final_list)
# helper funs to make df out of texts
# number of encyclicals of each pope

def get_df_of_enc(pope = 'all'):
    with open("pickles/selected_popes_names", "rb") as fp:   # Unpickling
        popes = pickle.load(fp)
    data_dict = {}
    if pope == 'all':
        # collect each popes encyclicas files and texts
        data_dict['popes']     = []
        data_dict['enc_title'] = []
        data_dict['enc_date']  = []
        data_dict['enc_text']  = []
        data_dict['enc_file']  = []
        for pope in popes:
            
            pope_dir = 'txts/' + pope
            all_files = os.listdir(pope_dir)
            enc_files = [file for file in all_files if file.startswith('enc_')]
            data_dict['enc_file'].append(enc_files)
            
            enc_titles = [filename[4:-13] for filename in enc_files]
            data_dict['enc_title'].append(enc_titles)
            
            enc_dates = [filename[-12:-4] for filename in enc_files]
            data_dict['enc_date'].append(enc_dates)
            
            enc_texts = []
            for filename in enc_files:
                with open(pope_dir+'/'+filename,encoding="utf-8") as infile:
                    contents = infile.read()
                enc_texts.append(contents)
            
            data_dict['enc_text'].append(enc_texts)
            
            
            data_dict['popes'].append(len(enc_files)*[pope])
            
        for key in data_dict.keys():
            data_dict[key] = flatten(data_dict[key])#list(itertools.chain.from_iterable(data_dict[key]))
            pass
    else:
        #todo if neccesary
        pass
    
    df = pd.DataFrame(data_dict)
    
    return(df)


def get_df_of_merged_enc(pope = 'all'):
    with open("pickles/selected_popes_names", "rb") as fp:   # Unpickling
        popes = pickle.load(fp)
    data_dict = {}
    if pope == 'all':
        # collect each popes encyclicas files and texts
        data_dict['popes']     = []
        data_dict['enc_file']  = []
        data_dict['enc_text']  = []
        
        for pope in popes:
            
            pope_dir = 'txts/' + pope
            all_files = os.listdir(pope_dir)
            enc_files = [file for file in all_files if file.startswith(pope)]
            data_dict['enc_file'].append(enc_files)
           
            enc_texts = []
            for filename in enc_files:
                with open(pope_dir+'/'+filename,encoding="utf-8") as infile:
                    contents = infile.read()
                enc_texts.append(contents)
            
            data_dict['enc_text'].append(enc_texts)
            
            
            data_dict['popes'].append(len(enc_files)*[pope])
            
        for key in data_dict.keys():
            data_dict[key] = flatten(data_dict[key])#list(itertools.chain.from_iterable(data_dict[key]))
            pass
    else:
        #todo if neccesary
        pass
    
    df = pd.DataFrame(data_dict)
    return(df)

def plot_difference_plotly(mdiff, title="", annotation=None):
    """Plot the difference between models.

    Uses plotly as the backend."""
    import plotly.graph_objs as go
    import plotly.offline as py

    annotation_html = None
    if annotation is not None:
        annotation_html = [
            [
                "+++ {}<br>--- {}".format(", ".join(int_tokens), ", ".join(diff_tokens))
                for (int_tokens, diff_tokens) in row
            ]
            for row in annotation
        ]

    data = go.Heatmap(z=mdiff, colorscale='RdBu', text=annotation_html)
    layout = go.Layout(width=950, height=950, title=title, xaxis=dict(title="topic"), yaxis=dict(title="topic"))
    py.iplot(dict(data=[data], layout=layout))