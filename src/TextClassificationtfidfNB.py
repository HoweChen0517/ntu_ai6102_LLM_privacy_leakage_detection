import os
from os import listdir, path
import math
import pandas as pd
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.utils import shuffle
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

def load_data(dataPath):
    with open(dataPath, 'r', encoding='utf-8') as f:
        data = f.readlines()

    data = [line.strip().split('\t') for line in data]
    data = pd.DataFrame(data, columns=['label', 'output'])

    return data

def trainTestSplit(data, test_Ratio=0.2, random_state=42):
    dataNum = data.shape[0]
    data = shuffle(data, random_state=random_state).reset_index(drop=True)
    trainNum = math.ceil(dataNum * (1 - test_Ratio))
    trainData = data.loc[:trainNum]
    testData = data.loc[trainNum:]
    return trainData, testData

def trainNBwithTFIDF(trainData):
    pipe = Pipeline(
        [
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=200)),
            ('svd', TruncatedSVD(n_components=100)),
            ('norm', Normalizer()),
            ('clf', CategoricalNB())
        ]
    )
    pipe.fit(trainData['output'], trainData['label'])
    return pipe

def GridSearchCVNBwithTFIDF(trainData,cv):
    pipe = Pipeline(
        [
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('svd', TruncatedSVD()),
            ('norm', Normalizer()),
            ('clf', CategoricalNB())
        ]
    )
    param_grid = {
        'tfidf__max_features': [100, 200, 500],
        'tfidf__norm': ['l1', 'l2', None],
        'tfidf__sublinear_tf': [True, False],
        'svd__n_components': [50, 100, 200],
        'clf__alpha': [0.1, 0.5, 1.0]
    }
    grid = GridSearchCV(estimator=pipe, param_grid=param_grid,
                        cv=cv, scoring='accuracy', n_jobs=cv)
    grid.fit(trainData['output'], trainData['label'])
    best_params = grid.best_params_

    pipe = Pipeline(
        [
            ('tfidf', TfidfVectorizer(stop_words='english', 
                                      max_features=best_params['tfidf__max_features'], 
                                      norm=best_params['tfidf__norm'], 
                                      sublinear_tf=best_params['tfidf__sublinear_tf'])),
            ('svd', TruncatedSVD(n_components=best_params['svd__n_components'])),
            ('norm', Normalizer()),
            ('clf', CategoricalNB())
        ]
    )

    return grid, pipe

if __name__ == '__main__':
    if os.getcwd().endswith('src'):
        ROOT_PATH = path.dirname(os.getcwd())
    else:
        ROOT_PATH = os.getcwd()

    edaFlag = False
    if edaFlag:
        data_path = path.join(ROOT_PATH, 'data', 'eda_train_data.txt')
        trainData = load_data(data_path)
        data_path = path.join(ROOT_PATH, 'data', 'test_data.txt')
        testData = load_data(data_path)
    else:
        data_path = path.join(ROOT_PATH, 'data', 'data.txt')
        data = load_data(data_path)
        trainData, testData = trainTestSplit(data, test_Ratio=0.2, random_state=42)

    cvFlag = True

    if edaFlag:
        if cvFlag:
            print('Naive Bayes + TFIDF + EDA + GridSearch:')
        else:
            print('Naive Bayes + TFIDF + EDA:')
    else:
        if cvFlag:
            print('Naive Bayes + TFIDF + GridSearch:')
        else:
            print('Naive Bayes + TFIDF:')

    if cvFlag:
        grid, pipe = GridSearchCVNBwithTFIDF(trainData, cv=5)
        print(f'best params:\n{grid.best_params_}')
        print(f'Best cross-validation score: {grid.best_score_:.3f}')
    else:
        pipe = trainNBwithTFIDF(trainData)

    pipe.fit(trainData['output'], trainData['label'])
    y_pred = pipe.predict(testData['output'])
    test_accuracy = accuracy_score(testData['label'], y_pred)
    result = classification_report(testData['label'], y_pred)

    print('Accuracy on test data:\n', test_accuracy)
    print('Classification report on test data:\n', result)

"""
Naive Bayes + TFIDF + GridSearch:
best params:
{'clf__alpha': 0.1, 'svd__n_components': 50, 'tfidf__max_features': 100, 'tfidf__norm': 'l1', 'tfidf__sublinear_tf': True}
Best cross-validation score: 0.501
Accuracy on test data:
 0.49166666666666664
Classification report on test data:
               precision    recall  f1-score   support

           0       0.49      1.00      0.66        59
           1       0.00      0.00      0.00        61

    accuracy                           0.49       120
   macro avg       0.25      0.50      0.33       120
weighted avg       0.24      0.49      0.32       120

====================================================================================================

Naive Bayes + TFIDF + EDA + GridSearch:
best params:
{'clf__alpha': 0.1, 'svd__n_components': 50, 'tfidf__max_features': 100, 'tfidf__norm': 'l1', 'tfidf__sublinear_tf': True}
Accuracy on test data:
 0.49166666666666664
Classification report on test data:
               precision    recall  f1-score   support

           0       0.49      1.00      0.66        59
           1       0.00      0.00      0.00        61

    accuracy                           0.49       120
   macro avg       0.25      0.50      0.33       120
weighted avg       0.24      0.49      0.32       120
"""