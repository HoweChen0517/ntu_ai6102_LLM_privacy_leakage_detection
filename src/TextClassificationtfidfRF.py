from os import listdir, path
import math
import pandas as pd
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
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
            ('clf', RandomForestClassifier())
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
            ('clf', RandomForestClassifier())
        ]
    )
    param_grid = {
        'tfidf__max_features': [100, 200, 500],
        'tfidf__norm': ['l1', 'l2', None],
        'tfidf__sublinear_tf': [True, False],
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
            ('tfidf', TfidfVectorizer(stop_words='english', 
                                      max_features=best_params['tfidf__max_features'], 
                                      norm=best_params['tfidf__norm'], 
                                      sublinear_tf=best_params['tfidf__sublinear_tf'])),
            ('svd', TruncatedSVD(n_components=best_params['svd__n_components'])),
            ('norm', Normalizer()),
            ('clf', RandomForestClassifier())
        ]
    )

    return pipe

if __name__ == '__main__':
    ROOT_PATH = ('/Users/howechen/Project/ntu_ai6102_LLM_privacy_leakage_detection')
    data_path = path.join(ROOT_PATH, 'data', 'data.csv')
    cvFlag = True

    data = pd.read_csv(data_path)

    trainData, testData = trainTestSplit(data, test_Ratio=0.2, random_state=42)

    if cvFlag:
        pipe = GridSearchCVNBwithTFIDF(trainData, cv=5)
    else:
        pipe = trainNBwithTFIDF(trainData)

    pipe.fit(trainData['output'], trainData['label'])
    y_pred = pipe.predict(testData['output'])
    test_accuracy = accuracy_score(testData['label'], y_pred)
    result = classification_report(testData['label'], y_pred)

    print('Accuracy on test data: ', result)
    print('='*50)
    print('Classification report on test data: ', result)
