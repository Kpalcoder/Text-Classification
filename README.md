<div align="center">

# **News Category Classification using spaCy Word Vectors**

</div>

## **Project Overview**

This project demonstrates a multi-class text classification pipeline to categorize news articles into one of three categories: **Business**, **Sports**, or **Crime**. The core of this project is the use of the powerful `spaCy` library, specifically its pre-trained GloVe word embeddings (`en_core_web_lg`), to convert text data into meaningful numerical vectors. Various machine learning models are then trained and evaluated to find the best classifier for this task.

---

## **Dataset**

The project utilizes the [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset), which contains news headlines and their corresponding short descriptions from HuffPost.

For this specific task, we focus on a subset of the data, filtering for articles in the following categories:
- **BUSINESS**
- **SPORTS**
- **CRIME**

---

## **Methodology**

The project follows a systematic approach from data preprocessing to model evaluation.

### **1. Data Loading and Filtering**
- The initial dataset is loaded from a JSON file.
- The data is filtered to retain only the 'BUSINESS', 'SPORTS', and 'CRIME' categories.

### **2. Data Balancing**
To prevent model bias towards the majority class, the dataset is balanced by down-sampling. All categories are reduced to match the size of the smallest category ('CRIME'), ensuring an equal number of samples for each class.

### **3. Text Preprocessing**
A custom preprocessing function using `spaCy` is applied to the `short_description` of each news article. This function performs two key steps:
- **Stop Word Removal:** Common words that do not add significant meaning (e.g., "the," "is," "a") are removed.
- **Lemmatization:** Words are converted to their base or root form (e.g., "running" becomes "run").

### **4. Text Vectorization with spaCy**
After preprocessing, the cleaned text is converted into high-dimensional numerical vectors.
- The `en_core_web_lg` model from spaCy, which contains 300-dimensional GloVe word embeddings, is used.
- For each document (short description), spaCy calculates the average of the token vectors, resulting in a single 300-dimensional vector that represents the semantic meaning of the text.

### **5. Model Training and Evaluation**
The vectorized data is split into training (80%) and testing (20%) sets. Several machine learning classifiers were trained and evaluated:
- Decision Tree
- Multinomial Naive Bayes (with `MinMaxScaler`)
- K-Nearest Neighbors (KNN)
- Random Forest
- Gradient Boosting

---

## **Results**

The performance of the models was evaluated based on accuracy and the classification report (precision, recall, f1-score). The Gradient Boosting Classifier achieved the highest accuracy.

### **Model Performance Summary**

| Model | Accuracy |
| :--- | :---: |
| Decision Tree | 56.0% |
| K-Nearest Neighbors | 65.1% |
| Multinomial Naive Bayes | 69.0% |
| Random Forest | 71.2% |
| **Gradient Boosting** | **71.6%** |

### **Best Model: Gradient Boosting Classifier**

The Gradient Boosting model provided the best and most balanced performance across all categories.

