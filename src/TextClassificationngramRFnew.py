import os
from os import listdir, path
import math
import pandas as pd
import warnings
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (classification_report, accuracy_score, 
                           confusion_matrix, roc_curve, auc,
                           precision_score, recall_score, f1_score)

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
            ('ngram', CountVectorizer(ngram_range=(1, 2),  # using 1-gram and 2-gram
                                    stop_words='english',
                                    max_features=200)),
            ('svd', TruncatedSVD(n_components=100)),
            ('norm', Normalizer()),
            ('clf', RandomForestClassifier())
        ]
    )
    param_grid = {
        # N-gram features
        'ngram__ngram_range': [(1, 2), (1, 3), (1, 4)],  # unigram, bigram, trigram
        'ngram__max_features': [1000, 2000, 3000],  # maximum number of features

        # SVD parameters
        'svd__n_components': [100, 200, 300],  # number of components

        # Random Forest parameters
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [10, 20, 30, 40, 50]
    }
    grid = GridSearchCV(estimator=pipe, param_grid=param_grid,
                       cv=cv, scoring='accuracy', n_jobs=cv)
    grid.fit(trainData['output'], trainData['label'])
    best_params = grid.best_params_

    pipe = Pipeline(
        [
            ('ngram', CountVectorizer(ngram_range=best_params['ngram__ngram_range'],
                                    stop_words='english',
                                    max_features=best_params['ngram__max_features']),),
            ('svd', TruncatedSVD(n_components=best_params['svd__n_components'])),
            ('norm', Normalizer()),
            ('clf', RandomForestClassifier(n_estimators=best_params['clf__n_estimators'],
                                         max_depth=best_params['clf__max_depth']))
        ]
    )

    return grid, pipe

def save_evaluation_metrics(y_true, y_pred, y_pred_proba, save_dir, prefix=''):
    """
    save evaluation metrics, confusion matrix and classification report to files
    
    Args:
        y_true: ground truth
        y_pred: predicted labels
        y_pred_proba: predicted probabilities
        save_dir: save directory
        prefix: file name prefix
    """
    # create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # calculate evaluation metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label='1'),
        'recall': recall_score(y_true, y_pred, pos_label='1'),
        'f1_score': f1_score(y_true, y_pred, pos_label='1')
    }
    
    # calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, f'{prefix}confusion_matrix_{timestamp}.png'))
    plt.close()
    
    # save metrics to JSON file
    with open(os.path.join(save_dir, f'{prefix}metrics_{timestamp}.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # save classification report to JSON file
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(os.path.join(save_dir, f'{prefix}classification_report_{timestamp}.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    return metrics

def plot_roc_curve(y_true, y_pred_proba, save_dir, prefix=''):
    """
    plot ROC curve and save the figure
    
    Args:
        y_true: ground truth
        y_pred_proba: predicted probabilities
        save_dir: save directory
        prefix: file name prefix
    """
    # create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # convert labels to binary
    y_true_num = np.array([1 if label == '1' else 0 for label in y_true])
    
    # calculate fpr, tpr and auc
    fpr, tpr, _ = roc_curve(y_true_num, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # save the figure
    plt.savefig(os.path.join(save_dir, f'{prefix}roc_curve_{timestamp}.png'))
    plt.close()
    
    # save roc data to JSON file
    roc_data = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'auc': roc_auc
    }
    with open(os.path.join(save_dir, f'{prefix}roc_data_{timestamp}.json'), 'w') as f:
        json.dump(roc_data, f, indent=4)
    
    return roc_auc

def evaluate_model(pipe, X_test, y_true, save_dir, prefix=''):
    """
    Evaluate the model and save evaluation results
    
    Args:
        pipe: trained model Pipeline
        X_test: test data
        y_true: ground truth
        save_dir: save directory
        prefix: file name prefix
    """
    # predict labels and probabilities
    y_pred = pipe.predict(X_test)
    y_pred_proba = pipe.predict_proba(X_test)
    
    # save evaluation metrics
    metrics = save_evaluation_metrics(y_true, y_pred, y_pred_proba, save_dir, prefix)
    
    # plot ROC curve
    auc_score = plot_roc_curve(y_true, y_pred_proba, save_dir, prefix)
    
    return metrics, auc_score

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

    prefix = 'rf_ngram_'    # save file prefix

    save_dir = os.path.join(ROOT_PATH, 'results', prefix)

    if edaFlag:
        prefix += 'eda_'
    if cvFlag:
        prefix += 'cv_'
    if not edaFlag and not cvFlag:
        prefix += 'base_'

    # metrics, auc_score = evaluate_model(
    #     pipe, 
    #     testData['output'], 
    #     testData['label'],
    #     save_dir,
    #     prefix
    # )
    # print('Results saved to:', save_dir)

    print(f'ngram parameters: {pipe.named_steps["ngram"].get_params()}')
    print(f'svd parameters: {pipe.named_steps["svd"].get_params()}')
    print(f'model parameters: {pipe.named_steps["clf"].get_params()}')
    
"""
RandomForest + N-gram + GridSearch:
best params:
{'clf__max_depth': 30, 'clf__n_estimators': 200, 'ngram__max_features': 1000, 'ngram__ngram_range': (1, 2), 'svd__n_components': 200}
Best cross-validation score: 0.929
Accuracy on test data:
 0.9166666666666666
Classification report on test data:
               precision    recall  f1-score   support

           0       0.95      0.88      0.91        59
           1       0.89      0.95      0.92        61

    accuracy                           0.92       120
   macro avg       0.92      0.92      0.92       120
weighted avg       0.92      0.92      0.92       120

ngram parameters: {'analyzer': 'word', 'binary': False, 'decode_error': 'strict', 'dtype': <class 'numpy.int64'>, 'encoding': 'utf-8', 'input': 'content', 'lowercase': True, 'max_df': 1.0, 'max_features': 1000, 'min_df': 1, 'ngram_range': (1, 2), 'preprocessor': None, 'stop_words': 'english', 'strip_accents': None, 'token_pattern': '(?u)\\b\\w\\w+\\b', 'tokenizer': None, 'vocabulary': None}
svd parameters: {'algorithm': 'randomized', 'n_components': 200, 'n_iter': 5, 'n_oversamples': 10, 'power_iteration_normalizer': 'auto', 'random_state': None, 'tol': 0.0}
model parameters: {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': 30, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 200, 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}
"""