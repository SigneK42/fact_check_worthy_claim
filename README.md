# fact_check_worthy_claim

## Description
This is the code repository accompanying the mandatory project in the course DAT550 - Data Mining and Deep Learning, University of Stavanger, Spring 2023.

## Requirements

Please note that if you're using the conda package manager, you may need to enable the conda-forge repo to install the required versions of the `pandas` and `fastparquet` packages.

    conda config --add channels conda-forge
    conda config --set channel_priority strict

Confirm that the conda-forge channel has been added and set to highest priority.

    conda config --show channels   

The output should look similar to this (notice that conda-forge is listed above the defaults channel):

```
channels:
  - conda-forge
  - defaults
```

Then, install the required packages from requirements.txt

    conda install --file requirements.txt

or 

    pip install -r requirements.txt

**Spacy**  
After installing the requirements you need to download the Spacy model:

    python -m spacy download en_core_web_sm
    

## Data Exploration
See the `data_exploration.ipynb` notebook.
Here we do initial data exploration, visualisations and correlation.

## Baseline Model
See the `baseline_model.ipynb` notebook.
Here we establish a baseline model using the scikit-learn RandomForestClassifier with default settings.

## Feature Generation
See the `feature_generation.ipynb` notebook.
Here we generate the features used in the classifiers.

## Random Forest Classifier
See the `RandomForest.ipynb` notebook.
Here we attempt improve on the RandomForestClassifier results by adding features and tuning parameters.

## SVM
See the `SVM.ipynb` notebook.
Here we use the SVM model to see if we can improve the results. We use the same features as created for the RandomForestClassifier, and also try tuning parameters for improved results.

