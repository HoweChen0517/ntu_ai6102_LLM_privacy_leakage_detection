# ntu_ai6102_LLM_privacy_leakage_detection

## Abstract
Large Language Models (LLMs) have greatly benefited nowadays people's life, providing AI applications ranging from conversational agents to automated content generation. However, they also pose risks of privacy leakage. Trained on extensive datasets, LLMs may inadvertently reproduce sensitive information, such as personal details or confidential data, leading to consequences like identity theft, financial loss, and erosion of trust in AI technologies. Addressing this requires effective methods to detect privacy leakage in LLM outputs.

This study aims to identify privacy-sensitive information in LLM-generated text using text classification. By analyzing latent semantic patterns in the outputs, we extract features with N-gram and TF-IDF methods, reduce dimensionality with Singular Value Decomposition (SVD), and classify with Logistic Regression, Support Vector Machine, and Random Forest. Evaluation metrics include Accuracy, Precision, Recall, F1-score, and AUC.

## Experiment Design
To perform classification, we choose 3 models, **LR**, **SVM** and **Random Forest (RF)**. Details about these models are mentioned above, they are well-suited for text classification and can handle binary classification tasks effectively. In our overall training pipeline, we add **SVD** and **Normalizer** to enhance feature representation and improve model performance.

Additionally, we use **GridSearchCV** to tune parameters for better classification performance. During the process, **GridSearchCV** trains the model on multiple subsets of the data and scores it on validation subsets, comparing the performance metrics (in our experiment we choose accuracy) across parameter combinations. By automating hyperparameter tuning, **GridSearchCV** makes it easier to optimize models and improve predictive accuracy while reducing the need for manual trial and error.