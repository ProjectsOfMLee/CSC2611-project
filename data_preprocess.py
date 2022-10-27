import contractions
import inflect
import nltk
import unicodedata
import json
import os
import numpy as np
#import umap.umap_ as umap

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from bs4 import BeautifulSoup as BS4
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")


# noise removal
def strip_html(text):
    soup = BS4(text, "html.parser")
    return soup.get_text()


def to_lower_case(text):
    return text.lower()


def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)


def remove_punctuation(text):

    for punct in "/-'":
        text = text.replace(punct, " ")

    for punct in "&":
        text = text.replace(punct, "and")

    for punct in "?!.,\"#$%'()*+-/:;<=>@[\\]^_`{|}~" + "“”’":
        text = text.replace(punct, "")
    return text


# tokenisation
def tokenise(text):
    words = word_tokenize(text)
    return words


# normalisation
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = (
            unicodedata.normalize("NFKD", word)
            .encode("ascii", "ignore")
            .decode("utf-8", "ignore")
        )
        new_words.append(new_word)
    return new_words

def remove_numbers(words):
    new_words=[]
    for word in words:
        if not(word.isdigit()):
            new_words.append(word)
    return new_words



def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    inflect_engine = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = inflect_engine.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words("english"):
            new_words.append(word)
    return new_words


def lemmatise_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatiser = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatiser.lemmatize(word, pos="v")
        lemmas.append(lemma)
    return lemmas


def preprocess_text(text):
    text = to_lower_case(text)

    # noise removal
    text = strip_html(text)
    text = replace_contractions(text)
    text = remove_punctuation(text)

    # tokenisation
    words = tokenise(text)

    # normalisation
    words = remove_non_ascii(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    words = lemmatise_verbs(words)

    return " ".join(words)


def get_sub(x, rev=False):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    if rev:
        res = x.maketrans("".join(sub_s), "".join(normal))
    else:
        res = x.maketrans("".join(normal), "".join(sub_s))
    return x.translate(res)


def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans("".join(normal), "".join(super_s))
    return x.translate(res)


def get_tail_from_data_path(data_path):
    return os.path.split(data_path)[-1].split(".")[0]


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def reduce_dimensions(
    vectors, compass_vectors=None, typ="tsne", output_dimensions=2, fit_on_compass=False
):
    if fit_on_compass is True:
        if typ == "tsne":
            raise NotImplementedError(
                f"'tsne' type not supported when `fit_on_compass` is set to 'True'."
            )
        if compass_vectors is None:
            raise ValueError(
                f"`compass_vectors` cannot be of type: {type(compass_vectors)} when `fit_on_compass` is set to 'True'."
            )

    if typ == "pca":
        pca = PCA(output_dimensions, random_state=42)
        if fit_on_compass:
            compass_embeddings = pca.fit_transform(compass_vectors)
            embeddings = pca.transform(vectors)
        else:
            embeddings = pca.fit_transform(vectors)
    elif typ == "tsne":
        tsne = TSNE(n_components=output_dimensions, init="pca", random_state=42)
        # compass_embeddings = tsne.fit_transform(compass_vectors)
        embeddings = tsne.fit_transform(vectors)
    
    #elif typ == "umap":
        #reducer = umap.UMAP(
            #n_components=output_dimensions, transform_seed=42, random_state=42)
        #if fit_on_compass:
            #compass_embeddings = reducer.fit_transform(compass_vectors)
            #embeddings = reducer.transform(vectors)
        #else:
            #embeddings = reducer.fit_transform(vectors)
    else:
        raise NotImplementedError(f"No implementation found for `typ`: {typ}.")
    return embeddings


def save_json(dict_obj, save_path):
    if save_path is not None:
        with open(save_path, "w") as json_f:
            json.dump(dict_obj, json_f, indent=4)


def save_npy(arr, save_path):
    if save_path is not None:
        with open(save_path, "wb") as npy_f:
            np.save(npy_f, arr)


def remove_keywords_util(remove_keywords_path, sorted_gram_count_mapping):
    with open(remove_keywords_path, "r") as f:
        removed_keywords = f.read().split(",")
    sorted_gram_count_mapping = {
        key: sorted_gram_count_mapping[key]
        for key in sorted_gram_count_mapping.keys()
        if key not in removed_keywords
    }
    return sorted_gram_count_mapping


def length_removed_keywords(remove_keywords_path):
    with open(remove_keywords_path, "r") as f:
        len_keywords = len(f.read().split(","))
    return len_keywords