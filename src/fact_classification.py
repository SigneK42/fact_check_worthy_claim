import pandas as pd
import string
import numpy as np

import nltk
from nltk import ngrams
from nltk import pos_tag
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger')

# Caching stopwords
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

#######################################################################


def data_loading():
    df_crowdsourced = pd.read_csv(
        r"https://zenodo.org/record/3609356/files/crowdsourced.csv?download=1"
    )  # should update this to read it directly from online source
    df_ground_truth = pd.read_csv(r"https://zenodo.org/record/3609356/files/groundtruth.csv?download=1")
    df = df_crowdsourced.append(df_ground_truth)  # should replace this with  pd.concat

    return df, df_crowdsourced, df_ground_truth


def data_qc(df):
    print(
        "The dataset contains ", sum(df.Sentence_id.duplicated()), "duplicated values"
    )
    print("The dataset contains ", df.isna().sum().sum(), " nan values")
    if df.isna().sum().sum() > 0:
        a = df.isna().sum() > 0
        print("The nan values are in the ", a[a].index[0], " column")
    return


def label_to_int(df):
    # can solve this more elegantly by doing the transform on all columns which are text
    # (except for Text, and the other one...)
    le = LabelEncoder()
    df.Speaker_party = le.fit_transform(df.Speaker_party)
    df.Speaker_title = le.fit_transform(df.Speaker_title)
    df.Speaker = le.fit_transform(df.Speaker)

    return df


def clean_text_manually(df):
    # convert all to lower case
    df.Text = df["Text"].str.lower()
    # remove punctuation
    translator = str.maketrans("", "", string.punctuation)
    df.Text = df.Text.str.translate(translator)
    # remove new line markers
    df = df.replace("\n", " ", regex=True)
    # remove the stop words
    df["Text"] = df.Text.apply(
        lambda x: " ".join([word for word in x.split() if word not in (stop_words)])
    )

    return df


def test_train_split(df):
    df["Year"] = df.File_id.str.split(".").str[0]
    df["Year"] = df.File_id.str.split("-").str[0].astype("int")
    df_test = df[df.Year > 2008].copy()
    df_train = df[df.Year <= 2008].copy()

    return df_train, df_test


###################################################################
### Functions for text processing #################################
###################################################################


def stem(df):
    # Create instance of stemmer
    stemmer = PorterStemmer()
    return (
        df.Text.str.split(" ")
        .apply(lambda x: [stemmer.stem(y) for y in x])
        .astype("str")
    )


def tfid(test, train, n_gram_range=1):
    vectorizer = TfidfVectorizer(
    ngram_range=(n_gram_range, n_gram_range)
    )
    train_vectorized = vectorizer.fit_transform(train)
    test_vectorized = vectorizer.transform(
        test
    )  # only using transform on test (not re-fitting)
    vocabulary = vectorizer.get_feature_names_out()

    return train_vectorized, test_vectorized, vocabulary



def tokenize(column):
    tokens = nltk.word_tokenize(column)
    return [w for w in tokens if w.isalpha()]

def get_pos(l):
    return str(list(zip(*l))[1])

def pos_tag_(df): #this function doesn't work yet
    df['tokenized'] = df.apply(lambda x: tokenize(x['Text']), axis=1)
    df['pos_tag'] = df.tokenized.apply(nltk.pos_tag)
    df.pos_tag = df.pos_tag.apply(lambda row: get_pos(row))
    return df['pos_tag']


##########################################
### Predicting ###########################
##########################################


def predict_it(
    train_feature,
    train_val,
    test_feature,
    method=RandomForestClassifier(max_depth = 20,
        random_state=42,
        class_weight="balanced_subsample",
        ),
    ):
    classifier = method
    classifier.fit(train_feature, train_val)

    return classifier.predict(train_feature), classifier.predict(test_feature)

def stupid_model(n_predictions = 2):
    return np.full(n_predictions,-1)


############################################
### Scoring predictions ####################
############################################


def score_it(test_true, test_pred, features = 'tfid', algorithm = 'RandomForrest'):

    # calculate all the different scores for each class and return as a dataframe?
    scores = pd.DataFrame(columns = ['alogrithm', 'features'], data = [[algorithm, features]])
    p, r, fs, s = precision_recall_fscore_support(test_true, test_pred, average = None, labels = [-1,0,1])
    pw, rw, fsw, sw = precision_recall_fscore_support(test_true, test_pred, average = 'weighted')

    # presicion NFS (non-factual sentence)
    scores['p_NFS'] = p[0]
    # presicion UFS (unimportant factual sentence)
    scores['p_UFS'] = p[1]
    # presicion CFS (check-worthy factual sentence)
    scores['p_CFS'] = p[2]
    # precision weighted average
    scores['p_wavg'] = pw

    # recall NFS (non-factual sentence)
    scores['r_NFS'] = r[0]
    # recall UFS (unimportant factual sentence)
    scores['r_UFS'] = r[1]
    # recall CFS (check-worthy factual sentence)
    scores['r_CFS'] = r[2]
    # precision weighted average
    scores['r_wavg'] = rw

    # fscore NFS (non-factual sentence)
    scores['f_NFS'] = fs[0]
    # fscore UFS (unimportant factual sentence)
    scores['f_UFS'] = fs[1]
    # fscore CFS (check-worthy factual sentence)
    scores['f_CFS'] = fs[2]
    # precision weighted average
    scores['f_wavg'] = fsw

    return scores

