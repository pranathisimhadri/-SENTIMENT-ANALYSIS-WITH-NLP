# SENTIMENT-ANALYSIS-WITH-NLP
COMPANY : CODTECH IT SOLUTIONS

NAME :Pranathi Simhadri

INTERN ID : CT04DM549

DOMAIN : MACHINE LEARNING

DURATION : 4 WEEEKS

MENTOR : NEELA SANTOSH


## 💬 Sentiment Analysis with NLP (TF-IDF + Logistic Regression)

This project demonstrates a simple yet effective **Sentiment Analysis** pipeline using **Natural Language Processing (NLP)** and **Logistic Regression**. The objective is to classify text data—such as reviews or comments—into **positive** or **negative** sentiment using only classical ML techniques.

The project is implemented in a Jupyter Notebook, making it easy to understand and experiment with. It focuses on **TF-IDF** for feature extraction and **Logistic Regression** for classification, offering a clean and lightweight solution for binary text classification problems.

---

## 🔍 Problem Statement

With the growing amount of online text data, identifying user sentiment has become a key need in business, media, and research. This project aims to classify raw text into sentiment labels using a traditional machine learning approach.

Rather than relying on complex deep learning models, this implementation shows how far you can go with basic tools when the preprocessing and modeling are done right.

---

## 🧠 Workflow Overview

### 1. **Data Loading and Inspection**

* Data is loaded using `pandas`.
* It is typically in a CSV format with two columns: text and sentiment label (e.g., "positive" or "negative").
* Initial exploration checks for class balance and missing values.

### 2. **Text Preprocessing**

Text cleaning is done using standard steps:

* Lowercasing
* Removing punctuation, symbols, and extra whitespace
* Optional: Removing stopwords

This step ensures the text is normalized before feature extraction.

### 3. **TF-IDF Vectorization**

* The cleaned text is transformed into numerical format using **TF-IDF (Term Frequency–Inverse Document Frequency)** from `scikit-learn`.
* TF-IDF gives more weight to words that are unique to a document and less weight to common words.
* The result is a sparse matrix that represents the importance of words across the dataset.

### 4. **Model Training (Logistic Regression)**

* A **Logistic Regression** model is trained on the TF-IDF vectors.
* The dataset is split into training and testing sets.
* The model is trained to predict whether the input text expresses a positive or negative sentiment.

### 5. **Model Evaluation**

* Model performance is evaluated using:

  * Accuracy
  * Precision
  * Recall
  * F1-Score
  * Confusion Matrix
* These metrics help understand how well the model generalizes to unseen data.

### 6. **Prediction**

* After training, the model can be used to predict the sentiment of new text inputs.
* Example predictions are shown in the notebook.

---

## ⚙️ Requirements

To run this notebook, install the following packages:

```bash
pip install pandas numpy scikit-learn nltk matplotlib
```

---

## 🚀 How to Run

1. Download or clone the repository.
2. Open the notebook in Jupyter or JupyterLab.
3. Run the cells in order:

   * Load and clean data
   * Transform with TF-IDF
   * Train and evaluate the model
   * Try custom predictions

---

## 📦 Applications

* Review classification (movies, products, services)
* Social media sentiment monitoring
* Simple feedback analysis

