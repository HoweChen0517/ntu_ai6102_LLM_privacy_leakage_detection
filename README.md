# ntu_ai6102_LLM_privacy_leakage_detection

## Abstract
Large Language Models (LLMs) have greatly benefited nowadays people's life, providing AI applications ranging from conversational agents to automated content generation. However, they also pose risks of privacy leakage. Trained on extensive datasets, LLMs may inadvertently reproduce sensitive information, such as personal details or confidential data, leading to consequences like identity theft, financial loss, and erosion of trust in AI technologies. Addressing this requires effective methods to detect privacy leakage in LLM outputs.

This study aims to identify privacy-sensitive information in LLM-generated text using text classification. By analyzing latent semantic patterns in the outputs, we extract features with N-gram and TF-IDF methods, reduce dimensionality with Singular Value Decomposition (SVD), and classify with Logistic Regression, Support Vector Machine, and Random Forest. Evaluation metrics include Accuracy, Precision, Recall, F1-score, and AUC.

## Data Source
Our dataset for privacy leakage detection consists of 600 Large Language Model (LLM) outputs, which were generated to address privacy concerns across six distinct privacy scenarios. Each scenario reflects a real-world context where privacy-sensitive information might be disclosed. The dataset was created in alignment with the team project guidelines, focusing on 6 main scenarios:
- **Personal Information Management** – Handling sensitive personal data such as social security numbers, banking information, and identifiers.
- **Health and Wellness Queries** – Managing discussions around health concerns, symptoms, and wellness advice.
- **Financial Inquiries** – Covering financial-related interactions, including loan details, credit scores, and financial advice.
- **Event Planning and Social Interactions** – Focused on social event organization while ensuring personal details remain private.
- **Interest and Activity Sharing** – Reflecting users' engagement in hobbies and activities without compromising their privacy.
- **Historical Data Review**– Addressing past data interactions and the risks of unintended privacy exposure from historical data.
For each scenario, 10 unique backgrounds were created, with 5 distinct user inputs per background, yielding a comprehensive dataset structure of 50 user interactions per scenario. Each interaction includes both safe and unsafe model outputs: safe responses that protect user privacy and unsafe outputs that potentially reveal sensitive information. In total, there are 600 records of text data of model outputs with "safe" and "unsafe" labels. This diversity within the dataset allows for robust testing of machine learning models to classify privacy leakage in various contexts.

## Experiment Design
To perform classification, we choose 3 models, **LR**, **SVM** and **Random Forest (RF)**. Details about these models are mentioned above, they are well-suited for text classification and can handle binary classification tasks effectively. In our overall training pipeline, we add **SVD** and **Normalizer** to enhance feature representation and improve model performance.

Additionally, we use **GridSearchCV** to tune parameters for better classification performance. During the process, **GridSearchCV** trains the model on multiple subsets of the data and scores it on validation subsets, comparing the performance metrics (in our experiment we choose accuracy) across parameter combinations. By automating hyperparameter tuning, **GridSearchCV** makes it easier to optimize models and improve predictive accuracy while reducing the need for manual trial and error.