import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import string
from itertools import cycle
from itertools import permutations, combinations, product
from nltk import ngrams
from nltk import pos_tag, pos_tag_sents
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, RocCurveDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

nltk.download('averaged_perceptron_tagger', quiet=True)
# Caching stopwords
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

nlp = spacy.load("en_core_web_sm")


#######################################################################


def data_loading(local=False):
    """Load dataset.

    Parameters
    ----------
    local : bool, optional
        Load dataset from local storage, by default False

    Returns
    -------
    Three Pandas.DataFrames; df, df_crowdsourced, df_ground_truth
        The first dataframe is the concatenation of the two last dataframes.
    """
    if local:
        df_crowdsourced = pd.read_csv("../data/crowdsourced.csv")
        df_ground_truth = pd.read_csv("../data/groundtruth.csv")
    else:
        df_crowdsourced = pd.read_csv(
            r"https://zenodo.org/record/3609356/files/crowdsourced.csv?download=1"
        )
        df_ground_truth = pd.read_csv(
            r"https://zenodo.org/record/3609356/files/groundtruth.csv?download=1"
        )
    df = pd.concat([df_crowdsourced, df_ground_truth]).reset_index(drop=True)
    return df, df_crowdsourced, df_ground_truth


def score_loading(fname='score'):
    """Load the score dataframes from csv files.

    Returns
    -------
    Two Pandas.DataFrames; score_train, score_test
        The dataframes with the scoring data for the train and test datasets.
    """
    score_train = pd.read_csv(f"../results/{fname}_train.csv", index_col=0)
    score_test = pd.read_csv(f"../results/{fname}_test.csv", index_col=0)
    return score_train, score_test


def score_saving(score_train, score_test, fname='score'):
    score_train.to_csv(f"../results/{fname}_train.csv")
    score_test.to_csv(f"../results/{fname}_test.csv")
    return


def data_qc(df):
    """Perform quality check of the data.

    Check if there are duplicated data, 
    and report if there are any NaN values.

    Parameters
    ----------
    df : Pandas.DataFrame
        Dataframe to be checked.
    """
    print(
        "The dataset contains ", sum(
            df.Sentence_id.duplicated()), "duplicated values"
    )
    print("The dataset contains ", df.isna().sum().sum(), " nan values")
    if df.isna().sum().sum() > 0:
        a = df.isna().sum() > 0
        print("The nan values are in the ", a[a].index[0], " column")
    return


def label_to_int(df):
    """_summary_

    Parameters
    ----------
    df : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # can solve this more elegantly by doing the transform on all columns which are text
    # (except for Text, and the other one...)
    le = LabelEncoder()
    df.Speaker_party = le.fit_transform(df.Speaker_party)
    df.Speaker_title = le.fit_transform(df.Speaker_title)
    df.Speaker = le.fit_transform(df.Speaker)
    return df


def clean_text(df, column='Text'):
    """Clean text.

    Convert text to lowercase, 
    remove stopwords, punctuatons and linebreaks.

    Parameters
    ----------
    df : Pandas.Dataframe
        Dataframe with text to be cleaned.

    Returns
    -------
    Pandas.DataFrame
        Dataframe with the cleaned text in 'cleaned_text' column.
    """
    # convert all to lower case
    df["cleaned_text"] = df[column].str.lower()
    # remove the stop words
    df["cleaned_text"] = df["cleaned_text"].apply(
        lambda x: " ".join(
            [word for word in x.split() if word not in (stop_words)])
    )
    # remove punctuation
    translator = str.maketrans("", "", string.punctuation)
    df["cleaned_text"] = df["cleaned_text"].str.translate(translator)
    # remove new line markers
    df["cleaned_text"] = df["cleaned_text"].replace("\n", " ", regex=True)
    return df


def test_train_split(df):
    """Split dataset in testing and training.

    The dataset is split according to the year. 
    Data after 2008 is used for the testing dataset, 
    while data up until and including year 2008 is used 
    for the training dataset.

    Parameters
    ----------
    df : Pandas.DataFrame
        The dataframe to be split.

    Returns
    -------
    Three dataframes; df_train, df_test, idx_train
        The first two dataframes contain the train and test datasets, 
        while the third dataframe contains the boolean index for the original dataframe. 
        True indicates training data, and False indicates testing data.
    """
    df["Year"] = df.File_id.str.split(".").str[0]
    df["Year"] = df.File_id.str.split("-").str[0].astype("int")
    idx_train = df['Year'] <= 2008
    df_train = df[idx_train].copy()
    df_test = df[~idx_train].copy()
    return df_train, df_test, idx_train


def to_latex(df):
    """Print Pandas dataframe in Latex format.

    Parameters
    ----------
    df : Pandas.DataFrame
        Dataframe to be converted
    """
    print(df.to_latex(
        index=False,
        float_format='%.3f',
        escape=True
    ))
    return

###################################################################
### Functions for text processing #################################
###################################################################


def stem(df, column='Text'):
    """Perform word stemming on the dataframe.

    Parameters
    ----------
    df : Pandas.DataFrame
        The dataframe with text to be stemmed.

    Returns
    -------
    Pandas.Dataframe
        The stemmed text
    """
    # Create instance of stemmer
    stemmer = PorterStemmer()
    return (
        df[column].str.split(" ")
        .apply(lambda x: [stemmer.stem(y) for y in x])
        .apply(lambda x: ' '.join(x))
    )


def tfid(train, test=None, n_gram_range=1, max_features=None):
    """Generate TF-IDF tokens.

    Generates TF-IDF tokens based on the text in the passed dataframe.
    If both train and test dataframes are passed, 
    then the tokens will be trained on the train dataset, 
    and the test dataset tokens will be generated from the trained 
    model without refitting to the test dataset.

    Parameters
    ----------
    train : Pandas.DataFrame
        Training dataset
    test : Pandas.DataFrame, optional
        Testing dataset, by default None
    n_gram_range : int, optional
        Size of N-grams to use, by default 1

    Returns
    -------
    Two (optional Three) Pandas.DataFrames
        Returns the same number of dataframes as passed.
        The dataframes contains the vectorised TF-IDF values.
        In addition the generated vocabulary is returned.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(n_gram_range, n_gram_range),
        max_features=max_features
    )
    train_vectorized = vectorizer.fit_transform(train)
    vocabulary = vectorizer.get_feature_names_out()
    if test is not None:
        test_vectorized = vectorizer.transform(test)
        return train_vectorized, test_vectorized, vocabulary
    else:
        return train_vectorized, vocabulary


def tokenize(column):
    """Tokenize words

    Parameters
    ----------
    column : array-like
        List or series with text

    Returns
    -------
    list
        list of tokens
    """
    tokens = nltk.word_tokenize(column)
    return [w for w in tokens if w.isalpha()]


def _get_pos(l):
    pos_list = []
    if len(l) > 0:
        pos_list = [w for w in list(zip(*l))[-1]]
    return pos_list


def get_pos_tags(column):
    """Generate POS-tags.

    Parameters
    ----------
    column : Pandas.Series
        Text to use for POS-tagging.

    Returns
    -------
    Pandas.Series
        The generated POS-tags.
    """
    tokens = column.apply(lambda x: tokenize(x))
    pos_tags = tokens.apply(nltk.pos_tag)
    pos_tags = pos_tags.apply(lambda row: _get_pos(row))
    pos_tags = pos_tags.apply(lambda x: ' '.join(x))
    return pos_tags


def ner_labels(column, batch_size=1000):
    """Generate NER-labels.

    Parameters
    ----------
    column : Pandas.Series
        The text to use for NER-label generation.
    batch_size : int, optional
        Batch size for the Spacy.pipe, by default 1000

    Returns
    -------
    List
        List with the identified NER-labels, 
        where the list index correspond to the 
        row index of the passed dataframe.
    """
    nlp = spacy.load("en_core_web_sm")
    entities_list = []
    for doc in nlp.pipe(column, batch_size=batch_size):
        entities = ' '.join([ent.label_ for ent in doc.ents])
        entities_list.append(entities)
    return entities_list

##########################################
### Predicting ###########################
##########################################


def predict_it(train_feature, train_val, test_feature,
               method=RandomForestClassifier(
                   random_state=42, class_weight="balanced_subsample"),
               ):
    """_summary_

    Parameters
    ----------
    train_feature : _type_
        _description_
    train_val : _type_
        _description_
    test_feature : _type_
        _description_
    method : _type_, optional
        _description_, by default RandomForestClassifier( random_state=42, class_weight="balanced_subsample", )

    Returns
    -------
    _type_
        _description_
    """
    classifier = method
    classifier.fit(train_feature, train_val)

    return classifier.predict(train_feature), classifier.predict(test_feature)


def dummy_model(n_predictions=2):
    """Dummy model.

    Returns array of specified length with all -1.

    Parameters
    ----------
    n_predictions : int, optional
        Size of array, by default 2

    Returns
    -------
    Numpy.array
        Array of -1's.
    """
    return np.full(n_predictions, -1)


############################################
### Scoring predictions ####################
############################################


def score_it(test_true, test_pred, features='W', algorithm='RFC'):
    """Generate scoring metrics.

    Parameters
    ----------
    test_true : Pandas.DataFrame
        True values
    test_pred : Pandas.DataFrame
        Predicted values
    features : str, optional
        Name of the features, by default 'W'
    algorithm : str, optional
        Name of the algorithm, by default 'RFC'

    Returns
    -------
    Pandas.DataFrame
        Dataframe with the calculated scoring metrics.
    """
    # calculate all the different scores for each class and return as a dataframe?
    scores = pd.DataFrame(columns=['algorithm', 'features'], data=[
                          [algorithm, features]])
    p, r, fs, s = precision_recall_fscore_support(
        test_true, test_pred, average=None, labels=[-1, 0, 1], zero_division=0)
    pw, rw, fsw, sw = precision_recall_fscore_support(
        test_true, test_pred, average='weighted', zero_division=0)

    # presicion NFS (non-factual sentence)
    scores['p_NFS'] = p[0].round(3)
    # presicion UFS (unimportant factual sentence)
    scores['p_UFS'] = p[1].round(3)
    # presicion CFS (check-worthy factual sentence)
    scores['p_CFS'] = p[2].round(3)
    # precision weighted average
    scores['p_wavg'] = pw.round(3)

    # recall NFS (non-factual sentence)
    scores['r_NFS'] = r[0].round(3)
    # recall UFS (unimportant factual sentence)
    scores['r_UFS'] = r[1].round(3)
    # recall CFS (check-worthy factual sentence)
    scores['r_CFS'] = r[2].round(3)
    # precision weighted average
    scores['r_wavg'] = rw.round(3)

    # fscore NFS (non-factual sentence)
    scores['f_NFS'] = fs[0].round(3)
    # fscore UFS (unimportant factual sentence)
    scores['f_UFS'] = fs[1].round(3)
    # fscore CFS (check-worthy factual sentence)
    scores['f_CFS'] = fs[2].round(3)
    # precision weighted average
    scores['f_wavg'] = fsw.round(3)

    return scores


############################################
### Running Experiments ####################
############################################

def run_experiment(clf, X_train, y_train, X_test, y_test, annotations):
    """Runs a single experiment for a classifier function.

    Trains the passed classifier on the training dataset, and calculates the test score using the testing dataset. 
    Then returns the trained classifier together with the training and testing scores.

    Parameters
    ----------
    clf : scikit-learn classifier
        Classifier that implements fit() and predict() functions.
    X_train : array-like
        Training features in a pandas dataframe or numpy array. 
        Format must be supported by the passed classifier.
    y_train : array-like
        Targets for the training dataset.
    X_test : array-like
        Testing features in a pandas dataframe of numpy array.
        Format must be supported by the passed classifier.
    y_test : array-like
        Targets for the testing dataset.
    annotations : dict
        Dictionary that must contain values for the keys `algorithm` and `features`.

    Returns
    -------
    clf, train_score, test_score
        Returns the trained classifier, training score dataframe, and testing score dataframe.
    """
    algorithm = annotations['algorithm']
    features = annotations['features']

    print(
        f'Running experiment with algorithm "{algorithm}" and features "{features}"')

    # Fit model
    clf.fit(X_train, y_train)

    # Print best parameters
    print(f'Best parameters found: {clf.best_params_}')

    # Generate predictions
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)

    # Get metrics
    train_score = score_it(y_train, pred_train,
                           algorithm=algorithm, features=features)
    test_score = score_it(
        y_test, pred_test, algorithm=algorithm, features=features)

    return clf, train_score, test_score

############################################
### Plotting Functions  ####################
############################################


def plot_train_test_score(df_score_train, df_score_test, method='SVM', order_by='f_wavg', stage='Testing'):
    """Plot train and test scores.

    Parameters
    ----------
    df_score_train : Dataframe
        Training results
    df_score_test : Dataframe
        Testing results
    method : str, optional
        Method used. Must be found in the "algorithm" column in the dataframe, by default 'SVM'
    order_by : str, optional
        Value to order by, by default 'f_wavg'
    stage : str, optional
        Stage to plot values from. Possible values: "Training" and "Testing", by default 'Testing'
    """
    # sort order
    sort_order = df_score_test[df_score_test['algorithm'] == method].sort_values(
        by=order_by)['features'].to_list()

    # merge dataframes
    df_score_all = pd.concat([
        df_score_train.assign(stage='Training'), 
        df_score_test.assign(stage='Testing')
        ])

    # convert to long format
    df_score_long = pd.melt(df_score_all, id_vars=[
                            'algorithm', 'features', 'stage'])

    # return pointplot
    fig, ax = plt.subplots(figsize=(6, 4))
    if method == 'compare':
        sort_order = df_score_test[df_score_test['algorithm'] == 'SVM'].sort_values(
            by=order_by)['features'].to_list()
        ax = sns.pointplot(df_score_long.query('variable=="%s" & stage=="%s"' % (order_by, stage)),
                           x='features', y='value', hue='algorithm', order=sort_order, ax=ax, markers='.')
    else:
        ax = sns.pointplot(df_score_long.query('algorithm=="%s" & variable=="%s"' % (method, order_by)),
                           x='features', y='value', hue='stage', order=sort_order, ax=ax)
    plt.setp(ax.lines, linewidth=1.5)
    ax.set_xticklabels(sort_order, rotation=60, ha='right')
    ax.set_xlabel('Features')
    ax.set_ylabel('Weighted f1-score')
    ax.grid()
    ax.legend()
    plt.show()

    return



def plot_train_time(data):
    """Plot model training times.

    Parameters
    ----------
    data : GridSearchCV
        Fitted instance of the GridSearchCV class
    """
    data = get_gridsearch_values(data)
    labels = data.index.unique()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax = sns.lineplot(data, x='Features', y='Mean_fit_time', ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha='right')
    ax.set_xlabel('Features')
    ax.set_ylabel('Seconds')
    ax.grid()
    plt.show()
    

def get_gridsearch_values(data):
    """Extract values from GridSearchCV for plotting.

    Parameters
    ----------
    data : GridSearchCV
        Fitted instance of the GridSearchCV class

    Returns
    -------
    Dataframe
        Dataframe with columns 'Features' and 'Mean_fit_time'
    """
    mean_fit_times = []
    features = []
    for key, value in data.items():
        fit_times = value.cv_results_['mean_fit_time']
        for fit_time in fit_times:
            features.append(key)
            mean_fit_times.append(fit_time)
    df = pd.DataFrame({'Features': features, 'Mean_fit_time': mean_fit_times})
    df.set_index('Features', inplace=True)
    sort_order = df.groupby('Features').mean().sort_values(by='Mean_fit_time').index.tolist()
    return df.loc[sort_order]


def prep_roc_data(y_onehot_test, y_score, n_classes):
    """Prepare data for ROC plotting function.
    
    Based on: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    Parameters
    ----------
    y_onehot_test : array-like of shape (n_samples, n_classes)
        True labels
    y_score : array-like of shape (n_samples, n_classes)
        Predicted probabilities
    n_classes : int
        Number of classes

    Returns
    -------
    fpr, tpr, roc_auc
        False positive rate, true positive rate, area under curve
    """
    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = dict(), dict(), dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    return fpr, tpr, roc_auc


def plot_roc_curve(y_true, y_score, target_names = ['NFS', 'UFS', 'CFS']):
    """Plot ROC curve.
    
    Based on: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels
    y_score : array-like of shape (n_samples, n_classes)
        Predicted probabilities. E.g. output from predict_proba() classifier function.
    target_names : list, optional
        List of class names. Must be in the same order as in `y_score`, by default ['NFS', 'UFS', 'CFS']
    """

    y_onehot_test = LabelBinarizer().fit_transform(y_true)
    n_classes = len(target_names)
    fpr, tpr, roc_auc = prep_roc_data(y_onehot_test, y_score, n_classes)

    fig, ax = plt.subplots(figsize=(6, 6))

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"ROC curve for {target_names[class_id]}",
            color=color,
            ax=ax,
        )

    plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
    plt.legend()
    plt.show()
