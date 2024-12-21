import os
from os import listdir, path
import math
import pandas as pd
import warnings

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
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

def trainRFwithNgram(trainData):
    pipe = Pipeline(
        [
            ('ngram', CountVectorizer(ngram_range=(1, 2),  # using 1-gram and 2-gram
                                    stop_words='english',
                                    max_features=200)),
            ('svd', TruncatedSVD(n_components=100)),
            ('norm', Normalizer()),
            ('clf', RandomForestClassifier())
        ]
    )
    pipe.fit(trainData['output'], trainData['label'])
    return pipe

def GridSearchCVRFwithNgram(trainData, cv):
    pipe = Pipeline(
        [
            ('ngram', CountVectorizer(stop_words='english')),
            ('svd', TruncatedSVD()),
            ('norm', Normalizer()),
            ('clf', RandomForestClassifier())
        ]
    )
    param_grid = {
        'ngram__max_features': [100, 200, 500],
        'ngram__ngram_range': [(1, 1), (1, 2), (1, 3)],  # try different n-gram
        'ngram__min_df': [1, 2, 3],  # minimum document frequency
        'svd__n_components': [50, 100, 200],
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [10, 20, 30, 40, 50]
    }
    grid = GridSearchCV(estimator=pipe, param_grid=param_grid,
                       cv=cv, scoring='accuracy', n_jobs=cv)
    grid.fit(trainData['output'], trainData['label'])
    best_params = grid.best_params_

    pipe = Pipeline(
        [
            ('ngram', CountVectorizer(stop_words='english',
                                    max_features=best_params['ngram__max_features'],
                                    ngram_range=best_params['ngram__ngram_range'],
                                    min_df=best_params['ngram__min_df'])),
            ('svd', TruncatedSVD(n_components=best_params['svd__n_components'])),
            ('norm', Normalizer()),
            ('clf', RandomForestClassifier(n_estimators=best_params['clf__n_estimators'],
                                         max_depth=best_params['clf__max_depth']))
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

    # record training settings
    if edaFlag:
        if cvFlag:
            print('RandomForest + N-gram + EDA + GridSearch:')
        else:
            print('RandomForest + N-gram + EDA:')
    else:
        if cvFlag:
            print('RandomForest + N-gram + GridSearch:')
        else:
            print('RandomForest + N-gram:')

    if cvFlag:
        grid, pipe = GridSearchCVRFwithNgram(trainData, cv=5)
        print(f'best params:\n{grid.best_params_}')
        print(f'Best cross-validation score: {grid.best_score_:.3f}')
    else:
        pipe = trainRFwithNgram(trainData)

    pipe.fit(trainData['output'], trainData['label'])
    y_pred = pipe.predict(testData['output'])
    test_accuracy = accuracy_score(testData['label'], y_pred)
    result = classification_report(testData['label'], y_pred)

    print('Accuracy on test data:\n', test_accuracy)
    print('Classification report on test data:\n', result)

"""
RandomForest + N-gram + GridSearch:
best params:
{'clf__max_depth': 20, 'clf__n_estimators': 100, 'ngram__max_features': 500, 'ngram__min_df': 2, 'ngram__ngram_range': (1, 3), 'svd__n_components': 200}
Accuracy on test data:
 0.9416666666666667
Classification report on test data:
               precision    recall  f1-score   support

           0       0.96      0.92      0.94        59
           1       0.92      0.97      0.94        61

    accuracy                           0.94       120
   macro avg       0.94      0.94      0.94       120
weighted avg       0.94      0.94      0.94       120

====================================================================================================

RandomForest + N-gram + EDA + GridSearch:
best params:
{'clf__max_depth': 50, 'clf__n_estimators': 300, 'ngram__max_features': 500, 'ngram__min_df': 3, 'ngram__ngram_range': (1, 3), 'svd__n_components': 200}
Accuracy on test data:
 0.925
Classification report on test data:
               precision    recall  f1-score   support

           0       0.98      0.86      0.92        59
           1       0.88      0.98      0.93        61

    accuracy                           0.93       120
   macro avg       0.93      0.92      0.92       120
weighted avg       0.93      0.93      0.92       120
"""