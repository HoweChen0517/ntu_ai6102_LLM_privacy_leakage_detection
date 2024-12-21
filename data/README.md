# Data Source

Our dataset for privacy leakage detection consists of 600 Large Language Model (LLM) outputs, which were generated to address privacy concerns across six distinct privacy scenarios. Each scenario reflects a real-world context where privacy-sensitive information might be disclosed. The dataset was created in alignment with the team project guidelines, focusing on 6 main scenarios:

- **Personal Information Management** – Handling sensitive personal data such as social security numbers, banking information, and identifiers.
- **Health and Wellness Queries** – Managing discussions around health concerns, symptoms, and wellness advice.
- **Financial Inquiries** – Covering financial-related interactions, including loan details, credit scores, and financial advice.
- **Event Planning and Social Interactions** – Focused on social event organization while ensuring personal details remain private.
- **Interest and Activity Sharing** – Reflecting users' engagement in hobbies and activities without compromising their privacy.
- **Historical Data Review**– Addressing past data interactions and the risks of unintended privacy exposure from historical data.

For each scenario, 10 unique backgrounds were created, with 5 distinct user inputs per background, yielding a comprehensive dataset structure of 50 user interactions per scenario. Each interaction includes both safe and unsafe model outputs: safe responses that protect user privacy and unsafe outputs that potentially reveal sensitive information. In total, there are 600 records of text data of model outputs with "safe" and "unsafe" labels. This diversity within the dataset allows for robust testing of machine learning models to classify privacy leakage in various contexts.