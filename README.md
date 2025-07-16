# Amazon Kindle Review Sentiment Classifier: NLP Benchmarking Project

## 📁 Project Summary
Built a robust sentiment classification system using Amazon Kindle reviews (up to 2014) by benchmarking multiple NLP vectorization techniques and machine learning classifiers. The primary goal was to identify the best combination of NLP encoding and classifier for the task of binary review sentiment classification.

---

## 📖 Dataset
- **Source:** Amazon Kindle Review Dataset
- **Link** https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
- **Task:** Binary classification
  - `rating < 3` → **0** (Negative)
  - `rating ≥ 3` → **1** (Positive)

---

## 🧹 Preprocessing Steps
- Lowercased all text
- Removed:
  - HTML tags (`BeautifulSoup`)
  - URLs
  - Stopwords (`nltk`)
  - Punctuation and non-alphanumeric characters
- Tokenized and lemmatized the words using `WordNetLemmatizer`

---

## 📝 NLP Vectorization Techniques Used
Compared four different NLP techniques:
1. **Bag of Words (BoW)**
2. **TF-IDF**
3. **Word2Vec (Custom trained, 300 dimensions)**
4. **Word2Vec (Pretrained Google News model)**

---

## 🧠 Baseline Classification
Used **Logistic Regression** across all four NLP techniques to establish baselines.

### 📊 Result:
**TF-IDF** provided the best performance on the test set.

---

## 🧪 Classifiers Benchmarked with TF-IDF
Evaluated 7 classifiers using the best-performing vectorizer (TF-IDF):

| Model                  | Notes                                |
|------------------------|--------------------------------------|
| 1. Logistic Regression | ✅ Best accuracy and interpretability |
| 2. Multinomial NB      | Fast but underperformed               |
| 3. Linear SVC          | Great results, scalable               |
| 4. Random Forest       | Decent, but slower                    |
| 5. XGBoost             | Strong but slightly overfit           |
| 6. K-Nearest Neighbors | Computationally expensive             |
| 7. SVC (non-linear)    | ❌ Inefficient for large datasets      |

### 🔹 Conclusion:
> **TF-IDF + Logistic Regression** performed the best in terms of accuracy, F1-score, and speed.

---

## 🛠️ Hyperparameter Tuning
Performed tuning on Logistic Regression using `GridSearchCV`:

- `C`: `[0.1, 1.0, 10]`
- `solver`: `['lbfgs', 'liblinear', 'saga']`
- `penalty`: `'l2'`

### 🔍 Outcome:
- `C=10` led to high training accuracy (~98%) but no improvement on test data
- Resulted in **overfitting**
- ✅ Final choice: **Default Logistic Regression (`C=1.0`)**

---

## ✅ Final Conclusion
- **Best Vectorizer:** TF-IDF
- **Best Classifier:** Logistic Regression (default)
- **Word2Vec (Google News)** underperformed TF-IDF on this dataset
- **Avoid using SVC** on large datasets; use **LinearSVC** instead

---

Feel free to explore or fork this project and test your own classifiers!
