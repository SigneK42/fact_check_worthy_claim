import pandas as pd
import string

import nltk
from nltk import ngrams

# Caching stopwords
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

#######################################################################


def data_loading():
    df_crowdsourced = pd.read_csv(r"data\crowdsourced.csv")
    df_ground_truth = pd.read_csv(r"data\groundtruth.csv")
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
        TfidfVectorizer, ngram_range=(n_gram_range, n_gram_range)
    )
    train_vectorized = vectorizer.fit_transform(train)
    test_vectorized = vectorizer.transform(
        test
    )  # only using transform on test (not re-fitting)

    return train_vectorized, test_vectorized


########################################################
### Predicting
##########################################


def predict_it(
    train_feature,
    train_val,
    test_feature,
    method=RandomForestClassifier(
        n_estimators=20,
        max_depth=20,
        random_state=42,
        class_weight="balanced_subsample",
    ),
):
    classifier = method
    classifier.fit(train_feature, train_val)

    return classifier.predict(train_feature), classifier.predict(test_feature)


############################################
### Scoring predictions
###############################################

def score_it(test_true, test_pred, train_true, train_pred):
    
    # calculate all the different scores for each class and return as a dataframe?

    

    return scores