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

def trainLRwithNgram(trainData):
    pipe = Pipeline(
        [
            ('ngram', CountVectorizer(ngram_range=(1, 2),  
                                    stop_words='english',
                                    max_features=200)),
            ('svd', TruncatedSVD(n_components=100)),
            ('norm', Normalizer()),
            ('clf', LogisticRegression(random_state=42))
        ]
    )
    pipe.fit(trainData['output'], trainData['label'])
    return pipe

def GridSearchCVLRwithNgram(trainData, cv):
    pipe = Pipeline(
        [
            ('ngram', CountVectorizer(stop_words='english')),
            ('svd', TruncatedSVD()),
            ('norm', Normalizer()),
            ('clf', LogisticRegression(random_state=42))
        ]
    )
    # parameter space
    param_grid = {
        # N-gram features
        'ngram__max_features': [1000, 2000, 3000],  # control the number of features
        'ngram__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 3)],  # add 2-gram and 3-gram
        'ngram__min_df': [2, 3, 5],  # adjust the minimum document frequency
        
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
            ('ngram', CountVectorizer(stop_words='english',
                                    max_features=best_params['ngram__max_features'],
                                    ngram_range=best_params['ngram__ngram_range'],
                                    min_df=best_params['ngram__min_df'])),
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
            print('LogisticRegression + N-gram + EDA + GridSearch:')
        else:
            print('LogisticRegression + N-gram + EDA:')
    else:
        if cvFlag:
            print('LogisticRegression + N-gram + GridSearch:')
        else:
            print('LogisticRegression + N-gram:')

    if cvFlag:
        grid, pipe = GridSearchCVLRwithNgram(trainData, cv=5)
        print(f'Best parameters:\n{grid.best_params_}')
        print(f'Best cross-validation score: {grid.best_score_:.3f}')
    else:
        pipe = trainLRwithNgram(trainData)

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
LogisticRegression + N-gram + GridSearch:
Best parameters:
{'clf__C': 10.0, 'clf__class_weight': None, 'clf__penalty': 'l2', 'clf__solver': 'saga', 'ngram__max_features': 3000, 'ngram__min_df': 2, 'ngram__ngram_range': (1, 3), 'svd__n_components': 200}
Best cross-validation score: 0.940

Accuracy on test data: 0.925

Classification report on test data:
               precision    recall  f1-score   support

           0       0.96      0.88      0.92        59
           1       0.89      0.97      0.93        61

    accuracy                           0.93       120
   macro avg       0.93      0.92      0.92       120
weighted avg       0.93      0.93      0.92       120

====================================================================================================

Logistics Regression + N-gram + EDA + GridSearch:
Best parameters:
{'clf__C': 1.0, 'clf__class_weight': None, 'clf__penalty': 'l2', 'clf__solver': 'liblinear', 'ngram__max_features': 3000, 'ngram__min_df': 5, 'ngram__ngram_range': (1, 3), 'svd__n_components': 300}
Best cross-validation score: 0.933

Accuracy on test data: 0.9

Classification report on test data:
               precision    recall  f1-score   support

           0       0.94      0.85      0.89        59
           1       0.87      0.95      0.91        61

    accuracy                           0.90       120
   macro avg       0.90      0.90      0.90       120
weighted avg       0.90      0.90      0.90       120
"""