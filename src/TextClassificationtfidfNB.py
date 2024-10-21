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
    ROOT_PATH = ('/workspaces/ntu_ai6102_LLM_privacy_leakage_detection')
    data_path = path.join(ROOT_PATH, 'data', 'data.csv')
    cvFlag = True

    data = pd.read_csv(data_path)

    trainData, testData = trainTestSplit(data, test_Ratio=0.2, random_state=42)

    if cvFlag:
        grid, pipe = GridSearchCVNBwithTFIDF(trainData, cv=5)
    else:
        pipe = trainNBwithTFIDF(trainData)

    print(f'best params:\n{grid.best_params_}')
    pipe.fit(trainData['output'], trainData['label'])
    y_pred = pipe.predict(testData['output'])
    test_accuracy = accuracy_score(testData['label'], y_pred)
    result = classification_report(testData['label'], y_pred)

    print('Accuracy on test data:\n', result)
    print('='*20,'After Grid Search, test result on best params:','='*20)
    print('Classification report on test data:\n', result)

"""
Accuracy on test data:
               precision    recall  f1-score   support

           0       0.50      1.00      0.67         8
           1       0.00      0.00      0.00         8

    accuracy                           0.50        16
   macro avg       0.25      0.50      0.33        16
weighted avg       0.25      0.50      0.33        16

==================== After Grid Search, test result on best params: ====================
Classification report on test data:
               precision    recall  f1-score   support

           0       0.50      1.00      0.67         8
           1       0.00      0.00      0.00         8

    accuracy                           0.50        16
   macro avg       0.25      0.50      0.33        16
weighted avg       0.25      0.50      0.33        16
"""