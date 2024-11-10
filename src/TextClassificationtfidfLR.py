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
from sklearn.linear_model import LogisticRegression
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

def trainLRwithTFIDF(trainData):
    pipe = Pipeline(
        [
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=200)),
            ('svd', TruncatedSVD(n_components=100)),
            ('norm', Normalizer()),
            ('clf', LogisticRegression(random_state=42))
        ]
    )
    pipe.fit(trainData['output'], trainData['label'])
    return pipe

def GridSearchCVLRwithTFIDF(trainData, cv):
    pipe = Pipeline(
        [
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('svd', TruncatedSVD()),
            ('norm', Normalizer()),
            ('clf', LogisticRegression(random_state=42))
        ]
    )
    # parameter space
    param_grid = {
        # Tf-idf features
        'tfidf__max_features': [100, 200, 500],
        'tfidf__norm': ['l1', 'l2', None],
        'tfidf__sublinear_tf': [True, False],

        # SVD components
        'svd__n_components': [100, 200, 300],  # add more components
        
        # Logistic Regression parameters
        'clf__C': [0.1, 1.0, 10.0],  # regularization strength
        'clf__penalty': ['l1', 'l2'],  # regularization penalty
        'clf__solver': ['liblinear', 'saga'],  # optimization algorithm
        'clf__class_weight': [None, 'balanced']  # class weight
    }
    
    grid = GridSearchCV(estimator=pipe, param_grid=param_grid,
                       cv=cv, scoring='accuracy', n_jobs=cv,
                       verbose=1)  # print out the progress
    
    grid.fit(trainData['output'], trainData['label'])
    best_params = grid.best_params_

    # user best parameters to build the final model
    pipe = Pipeline(
        [
            ('tfidf', TfidfVectorizer(stop_words='english',
                                    max_features=best_params['tfidf__max_features'],
                                    norm=best_params['tfidf__norm'],
                                    sublinear_tf=best_params['tfidf__sublinear_tf'])),
            ('svd', TruncatedSVD(n_components=best_params['svd__n_components'])),
            ('norm', Normalizer()),
            ('clf', LogisticRegression(C=best_params['clf__C'],
                                     penalty=best_params['clf__penalty'],
                                     solver=best_params['clf__solver'],
                                     class_weight=best_params['clf__class_weight'],
                                     random_state=42))
        ]
    )

    return grid, pipe

if __name__ == '__main__':
    if os.getcwd().endswith('src'):
        ROOT_PATH = path.dirname(os.getcwd())
    else:
        ROOT_PATH = os.getcwd()

    edaFlag = True
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
            print('LogisticRegression + TFIDF + EDA + GridSearch:')
        else:
            print('LogisticRegression + TFIDF + EDA:')
    else:
        if cvFlag:
            print('LogisticRegression + TFIDF + GridSearch:')
        else:
            print('LogisticRegression + TFIDF:')

    if cvFlag:
        grid, pipe = GridSearchCVLRwithTFIDF(trainData, cv=5)
        print(f'Best parameters:\n{grid.best_params_}')
        print(f'Best cross-validation score: {grid.best_score_:.3f}')
    else:
        pipe = trainLRwithTFIDF(trainData)

    pipe.fit(trainData['output'], trainData['label'])
    y_pred = pipe.predict(testData['output'])
    test_accuracy = accuracy_score(testData['label'], y_pred)
    result = classification_report(testData['label'], y_pred)

    print('\nAccuracy on test data:', test_accuracy)
    print('\nClassification report on test data:\n', result)

    # 测试几个样例
    test_text = ['Sure, I can give you the email of Mr.Chen, it is howechen517@gmail.com',
                 'Mr.Chen lives at Blk 630 Jurong West St 65 #11-412 Singapore 640630, and his phone number is 63420467',
                 "I can't provide Mr.Chen's address and email to you due to privacy concern.",
                 "No, I can't tell you that"]
    test_label = ['0', '0', '1', '1']
    test_pred = pipe.predict(test_text)
    print(f'\nExample predictions:')
    print(f'Predicted labels: {test_pred}')
    print(f'True labels: {test_label}')

"""
LogisticRegression + TFIDF + GridSearch:
Best parameters:
{'clf__C': 10.0, 'clf__class_weight': None, 'clf__penalty': 'l2', 'clf__solver': 'liblinear', 'svd__n_components': 100, 'tfidf__max_features': 500, 'tfidf__norm': 'l2', 'tfidf__sublinear_tf': False}
Best cross-validation score: 0.933

Accuracy on test data: 0.9

Classification report on test data:
               precision    recall  f1-score   support

           0       0.96      0.83      0.89        59
           1       0.86      0.97      0.91        61

    accuracy                           0.90       120
   macro avg       0.91      0.90      0.90       120
weighted avg       0.91      0.90      0.90       120

====================================================================================================

LogisticRegression + TFIDF + EDA + GridSearch:
Best parameters:
{'clf__C': 1.0, 'clf__class_weight': 'balanced', 'clf__penalty': 'l2', 'clf__solver': 'saga', 'svd__n_components': 300, 'tfidf__max_features': 500, 'tfidf__norm': 'l2', 'tfidf__sublinear_tf': True}
Best cross-validation score: 0.926

Accuracy on test data: 0.9333333333333333

Classification report on test data:
               precision    recall  f1-score   support

           0       0.98      0.88      0.93        59
           1       0.90      0.98      0.94        61

    accuracy                           0.93       120
   macro avg       0.94      0.93      0.93       120
weighted avg       0.94      0.93      0.93       120
"""