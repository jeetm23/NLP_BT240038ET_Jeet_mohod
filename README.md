# SMS Spam Detection using NLP and Machine Learning

## Student Information

| Field | Details |
|---|---|
| **Student Name** | Jeet Mohod |
| **Roll Number** | BT240038ET |

---

## Problem Statement / Objective

The objective of this project is to develop an SMS Spam Detection system that automatically classifies incoming SMS messages as **Spam** or **Ham** (legitimate) using Natural Language Processing (NLP) and Machine Learning techniques. The system aims to filter out unwanted promotional and fraudulent messages from genuine communication, thereby improving user experience and security.

---

## Introduction

SMS (Short Message Service) spam refers to unsolicited messages sent in bulk to mobile phone users. These messages typically contain advertisements, fraudulent schemes, phishing links, or prize-winning scams. With the increasing volume of spam messages, manual filtering becomes impractical and time-consuming.

This project leverages **Natural Language Processing (NLP)** techniques combined with **Machine Learning** algorithms to build an intelligent system that can automatically detect and classify SMS messages. The system processes raw text through a multi-stage NLP pipeline including text cleaning, tokenization, stopword removal, stemming, and TF-IDF vectorization before applying a **Multinomial Naive Bayes** classifier for prediction.

The project includes both a Python-based backend for model training and evaluation, and an interactive web-based frontend for real-time SMS classification.

---

## NLP Techniques Used

The following NLP techniques are applied in this project:

1. **Text Cleaning** — Conversion to lowercase, removal of special characters, numbers, and extra whitespace to normalize the raw text data.

2. **Tokenization** — Splitting the cleaned text into individual words (tokens) for further processing.

3. **Stopword Removal** — Filtering out common English words (e.g., "the", "is", "and") that do not contribute to the classification task using the NLTK stopwords corpus.

4. **Stemming** — Reducing words to their root/base form using the **Porter Stemmer** algorithm (e.g., "running" becomes "run", "messages" becomes "messag").

5. **TF-IDF Vectorization** — Converting preprocessed text into numerical feature vectors using **Term Frequency-Inverse Document Frequency (TF-IDF)**. This method assigns higher weights to words that are frequent in a specific document but rare across the entire corpus, making it effective for text classification.

6. **Multinomial Naive Bayes Classification** — A probabilistic classifier based on Bayes' theorem, particularly suited for text classification tasks with TF-IDF features. It calculates the probability of a message being spam or ham based on the word frequencies.

---

## Dataset / Source of Data

| Property | Details |
|---|---|
| **Dataset Name** | UCI SMS Spam Collection |
| **Source** | UCI Machine Learning Repository |
| **URL** | https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection |
| **Total Messages** | 5,572 |
| **Ham Messages** | 4,825 (86.6%) |
| **Spam Messages** | 747 (13.4%) |
| **Format** | Tab-separated values (TSV) |
| **Language** | English |

The dataset consists of SMS messages labeled as either "ham" (legitimate) or "spam" (unsolicited). It was collected from free sources on the internet and is widely used for SMS spam detection research.

---

## Software / Tools / Libraries Used

| Software / Library | Purpose |
|---|---|
| **Python 3.13** | Core programming language |
| **scikit-learn** | Machine Learning model (Naive Bayes), TF-IDF Vectorizer, train-test split, evaluation metrics |
| **NLTK** | Natural Language Processing — stopwords, tokenization, stemming |
| **Pandas** | Data loading, manipulation, and analysis |
| **NumPy** | Numerical computations |
| **Matplotlib** | Data visualization — confusion matrix, distribution plots |
| **Seaborn** | Statistical data visualization |
| **HTML5 / CSS3 / JavaScript** | Frontend web interface |
| **Git / GitHub** | Version control and code hosting |

---

## Methodology / Workflow

The project follows a systematic NLP pipeline approach:

```
Step 1: Data Collection
    |
    v
Step 2: Data Exploration & Analysis
    |
    v
Step 3: Text Preprocessing (NLP Pipeline)
    |-- Lowercase conversion
    |-- Special character removal
    |-- Tokenization
    |-- Stopword removal
    |-- Stemming (Porter Stemmer)
    |
    v
Step 4: Feature Extraction (TF-IDF Vectorization)
    |-- Unigrams and Bigrams
    |-- Max 5000 features
    |
    v
Step 5: Train-Test Split (80:20 ratio)
    |
    v
Step 6: Model Training (Multinomial Naive Bayes)
    |
    v
Step 7: Model Evaluation
    |-- Accuracy, Precision, Recall, F1-Score
    |-- Confusion Matrix
    |
    v
Step 8: Model Saving (Pickle serialization)
    |
    v
Step 9: Prediction / Interactive Demo
```

---

## Steps to Execute the Program / Project

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Web browser (for frontend UI)

### Step 1: Clone the Repository

```bash
git clone https://github.com/jeetm23/NLP_BT240038ET_Jeet_mohod.git
cd NLP_BT240038ET_Jeet_mohod
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Spam Detection Script

```bash
python spam_detection.py
```

This will:
- Load the SMS dataset
- Preprocess text using NLP pipeline
- Train the Multinomial Naive Bayes model
- Evaluate and display metrics
- Generate visualization plots
- Save the trained model
- Launch an interactive demo

### Step 4: Open the Web Interface

Open `index.html` in any web browser to use the interactive SMS spam detection UI.

---

## Sample Input

### Example 1 — Spam Message

```
Input: "Congratulations! You have won a free prize. Click here to claim now."
```

### Example 2 — Ham (Legitimate) Message

```
Input: "Hi, are you coming to college today?"
```

### Example 3 — Spam Message

```
Input: "WINNER!! You've been selected to receive a $900 prize reward! Call 09061701461 now."
```

### Example 4 — Ham (Legitimate) Message

```
Input: "Can you send me the notes from yesterday's lecture?"
```

---

## Sample Output

### For Spam Message:

```
Input:  "Congratulations! You have won a free prize. Click here to claim now."

Result:     SPAM
Confidence: 96.4%
Ham Prob:   3.6%
Spam Prob:  96.4%

Key Indicators: Prize/Win claim, Free offer, Congratulations hook, Call-to-action
```

### For Ham Message:

```
Input:  "Hi, are you coming to college today?"

Result:     HAM
Confidence: 98.1%
Ham Prob:   98.1%
Spam Prob:  1.9%

Analysis: No spam indicators detected. Natural conversational tone.
```

---

## Results / Observations

### Model Performance Metrics

| Metric | Score |
|---|---|
| **Training Accuracy** | 98.24% |
| **Test Accuracy** | 97.13% |
| **Precision (Spam)** | 100% |
| **Recall (Spam)** | 83.89% |
| **F1-Score (Spam)** | 91.24% |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Ham | 0.96 | 1.00 | 0.98 | 965 |
| Spam | 1.00 | 0.84 | 0.91 | 150 |

### Key Observations

1. The **Multinomial Naive Bayes** model achieves high accuracy (97%) on the test dataset, demonstrating its effectiveness for SMS spam classification.

2. **Precision for spam is 100%**, meaning when the model predicts a message as spam, it is always correct. This minimizes false positives (legitimate messages wrongly classified as spam).

3. **Recall for spam is 84%**, indicating some spam messages are misclassified as ham. This is acceptable as it prioritizes not blocking legitimate messages.

4. **TF-IDF with bigrams** improves classification by capturing two-word patterns common in spam (e.g., "free prize", "click here", "call now").

5. Spam messages tend to be **longer** and contain more **capital letters**, **phone numbers**, and **monetary references** compared to ham messages.

6. The dataset is **imbalanced** (86.6% ham, 13.4% spam), but the model handles this well due to the Naive Bayes algorithm's robustness with skewed data.

---

## Conclusion

This project successfully demonstrates the application of **Natural Language Processing** and **Machine Learning** for SMS spam detection. The system processes raw text through a comprehensive NLP pipeline — text cleaning, tokenization, stopword removal, stemming, and TF-IDF vectorization — before classifying messages using a **Multinomial Naive Bayes** model.

The model achieves a **test accuracy of 97%** with **100% precision** for spam detection, making it reliable for real-world use. The web-based interface provides an intuitive way to interact with the model and visualize the classification results.

Key takeaways from this project:
- NLP preprocessing significantly improves classification accuracy
- TF-IDF vectorization effectively captures important text features
- Naive Bayes is a simple yet powerful algorithm for text classification
- The combination of NLP and ML provides an efficient solution for spam filtering

---

## Resources / References

1. **UCI SMS Spam Collection Dataset**
   - Almeida, T.A., Gomez Hidalgo, J.M., Yamakami, A. Contributions to the Study of SMS Spam Filtering: New Collection and Results. Proceedings of the 2011 ACM Symposium on Document Engineering.
   - URL: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

2. **scikit-learn Documentation**
   - Multinomial Naive Bayes: https://scikit-learn.org/stable/modules/naive_bayes.html
   - TF-IDF Vectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

3. **NLTK Documentation**
   - Natural Language Toolkit: https://www.nltk.org/
   - Stopwords Corpus: https://www.nltk.org/nltk_data/

4. **Text Classification with Python**
   - Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O'Reilly Media Inc.

5. **Naive Bayes for Text Classification**
   - Manning, C.D., Raghavan, P., Schutze, H. (2008). Introduction to Information Retrieval. Cambridge University Press. Chapter 13: Text Classification.

6. **Python Documentation**
   - https://docs.python.org/3/

---

*Project completed by Jeet Mohod (BT240038ET) as part of the NLP Application Based Mini Project.*
