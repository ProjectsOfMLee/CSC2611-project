from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import os
import pandas as pd
from nltk.util import ngrams
from nltk import FreqDist
from collections import Counter
from nltk.tokenize import sent_tokenize
from collections import defaultdict
import json
from data_preprocess import *
import pickle
from gensim.parsing.preprocessing import remove_stopwords
from tqdm import tqdm,trange

#augment_data = pickle.load(open("extra_s2orc.pickle","rb"))
#for yr in range(1994,2022):
    #Data_Augmentation(year=yr)

def Build_mapping(year:str):
    '''
    Build index to word and word to index map for given year
    If year is not available, build map for entire corpus
    '''
    target_files = [x for x in os.listdir(data_path)]
    index2word={}
    word2index={}
    if year == "all":
        for f in target_files:
            temp = open(data_path+f,"r",encoding='utf-8').read()
            for word in temp.split():
                if word not in word2index:
                    word2index[word] = len(word2index)
                    index2word[len(word2index)-1] = word
    else:
        temp = open(data_path+f"{year}.txt","r",encoding='utf-8').read()
        for word in temp.split():
            if word not in word2index:
                word2index[word] = len(word2index)
                index2word[len(word2index)-1] = word

    return index2word, word2index

def Build_freq_dict(year:str):
    '''
    Build frequency dictionary for given year
    '''
    f = open(data_path+f"{year}.txt","r",encoding='utf-8').read()
    unigram = FreqDist(f.split())
    return unigram


def Get_Unigram(word:str, year:str):
    '''
    return frequency of a word in text from a given year
    '''
    f = open(data_path+f"{year}.txt","r",encoding='utf-8').read()
    unigram = FreqDist(f.split())
    return unigram[word] if word in unigram else 0

def Get_topk(k:int, year:str):
    '''
    Get top k common words in the text from given year
    '''
    f = open(data_path+f"{year}.txt","r",encoding='utf-8').read()
    freq = FreqDist(f.split())
    return freq.most_common(k)

def Get_sentences(year:str):
    '''
    get sentences from unprocessed data for a given year
    '''
    data = json.load(open(data_path+'unprocessed.json'))
    abstract = ""
    for record in data[year]:
        abstract += record["abstract"]
    sentences = sent_tokenize(abstract)
    
    for i in range(len(sentences)):
        sentences[i] = sentences[i].replace("\n"," ")
        temp = remove_punctuation(sentences[i])
        temp = to_lower_case(temp)
        temp = remove_stopwords(temp.split())
        temp = lemmatise_verbs(temp)
        temp = remove_numbers(temp)
        sentences[i] = " ".join(temp)
        
    for s in sentences:    
        if (not s) or (len(s.split())<2):
            sentences.remove(s)
    return sentences


def Data_Augmentation(year:int):

    file1 = open(f"./data/{year}.txt")
    lines = file1.read()
    extra = " ".join(augment_data[year])
    extra = remove_punctuation(extra)
    extra = to_lower_case(extra)
    extra = remove_stopwords(extra)
    extra = lemmatise_verbs(extra)
    extra = remove_numbers(extra)
    #lines += " "
    lines += ("".join(extra))
    out_file = open(f"./data_augmented/{year}.txt", "w",encoding='utf-8')
    out_file.write(lines)
    
    file1.close()
    out_file.close()

def Get_common_words(threshold):

    '''
    Get common words across all years and filter the words that have frequency more than threhold
    '''
    common_words = set([k for k, v in Build_freq_dict(year="1994").items() if v > threshold])
    for yr in trange(1995,2021):
        
        temp = [k for k, v in Build_freq_dict(year=f"{yr}").items() if v > threshold]
        common_words.intersection_update(temp)
    
    common_words = sorted(common_words)
    return common_words

def co_occurrence(sentences, window_size):
    '''
    Get co-occurence of words from string of sentences within the window_size
    '''
    d = defaultdict(int)
    vocab = set()

    text = sentences.split()
    for i in trange(len(text)):
        token = text[i]
        vocab.add(token)  
        next_token = text[i+1 : i+1+window_size]
        for t in next_token:
            key = tuple(sorted([t, token]))
            d[key] += 1
            
    vocab = sorted(vocab)  
    df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                      index=vocab,
                      columns=vocab)
    for key, value in d.items():
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value
    
    df=df[list(common_words)]
    df=df.loc[df.index.isin(common_words)]
    
    return df

def construct_ppmi(df):
    '''
    construct ppmi based on co-occurence dataframe
    '''
    col_totals = df.sum(axis=0)
    total = col_totals.sum()
    row_totals = df.sum(axis=1)
    expected = np.outer(row_totals, col_totals) / total
    df = df / expected
    with np.errstate(divide='ignore'):
        df = np.log(df)
    df[np.isinf(df)] = 0.0  
    df[df < 0] = 0.0
    
    return df

def get_ppmi_embedding(year,window_size):
    '''
    get ppmi embedding for given word for the text in given year
    '''
    text = open(data_path+f"{year}.txt","r",encoding='utf-8').read()
    df_cooccurrence = co_occurrence(text, window_size=4)

    #from IPython.display import display, HTML
    #display(HTML(df_cooccurrence.to_html()))

    df_ppmi=construct_ppmi(df_cooccurrence).sort_index(ascending=True)
    df_ppmi = df_ppmi.reindex(sorted(df_ppmi.columns), axis=1)
    return df_ppmi

def get_nearest_neighbor(word,ppmi_matrix,k):
    # This is a helper function which gets k nearest neighbor words
    if (word not in common_words):
        return []
    
    matrix_year = []
    for wd in common_words:
        matrix_year.append(list(ppmi_matrix[wd]))
    matrix_year = np.array(matrix_year)

    knn_temp = NearestNeighbors(n_neighbors=k)
    knn_temp.fit(matrix_year)

    vector_temp = np.asarray(list(ppmi_matrix[word])).reshape(1, -1)
    neighbor_idx = knn_temp.kneighbors(vector_temp,k,return_distance=True)
    #print(neighbor_idx)
    
    ret = [common_words[i] for i in list(neighbor_idx)[0]]
    ret.remove(word)
    return ret

if __name__ == "__main__":
    common_words=Get_common_words(threshold=20)
    print(common_words)
    print(get_nearest_neighbor(word="deep",ppmi_matrix=ppmi_1994,k=6))
    
    