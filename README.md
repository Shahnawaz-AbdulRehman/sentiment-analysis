# Sentiment Analysis Project: README

## Overview
This project performs sentiment analysis on IMDB movie reviews. It uses various preprocessing techniques to clean text data, implements feature extraction methods like Bag of Words (BoW) and TF-IDF, and applies machine learning models such as Logistic Regression and Support Vector Machines (SVM) to classify sentiments. A Flask application is also provided for user interaction.

## Features
- **Preprocessing:** Text cleaning, tokenization, stemming, and removal of stopwords.
- **Feature Extraction:** Bag of Words and TF-IDF.
- **Machine Learning Models:** Logistic Regression and SVM.
- **Web Interface:** Flask-based application to input reviews and predict sentiment.
- **Model Serialization:** Save and reuse trained models and vectorizers.

---

## Requirements

### Libraries
Ensure the following Python libraries are installed:
- `numpy`
- `pandas`
- `seaborn`
- `nltk`
- `scikit-learn`
- `bs4` (BeautifulSoup)
- `Flask`

Install them via pip:
```bash
pip install numpy pandas seaborn nltk scikit-learn beautifulsoup4 flask
```

### Data
The project expects a CSV file named `IMDB Dataset.csv` with the following columns:
- **review**: Movie review text.
- **sentiment**: Sentiment labels (`Positive` or `Negative`).

---

## Getting Started

### Step 1: Data Preprocessing
The script cleans and preprocesses the data by:
- Removing HTML tags, special characters, and stopwords.
- Applying stemming.
- Splitting data into training and testing sets.

### Step 2: Feature Extraction
Two feature extraction techniques are applied:
1. Bag of Words (BoW)
2. TF-IDF

### Step 3: Model Training
The Logistic Regression and SVM models are trained on both feature sets:
- BoW features: `cv_train_reviews` and `cv_test_reviews`.
- TF-IDF features: `tv_train_reviews` and `tv_test_reviews`.

### Step 4: Model Evaluation
The models are evaluated using metrics such as accuracy score and classification report.

### Step 5: Save Models and Vectorizers
Trained models and vectorizers are saved as `.pkl` files using the `pickle` library.

---

## Flask Web Application

### File: `app.py`
The Flask application provides:
- A web interface to input text or upload a file.
- Sentiment prediction for the input data using the trained Logistic Regression model on BoW features.

### Run the Flask App
```bash
python app.py
```

### Access the Web Interface
Visit `http://127.0.0.1:5000/` in your browser.

---

## Project Structure

```
.
├── IMDB Dataset.csv         # Input dataset
├── preprocess.py            # Data preprocessing logic
├── train_model.py           # Model training and evaluation
├── app.py                   # Flask application
├── SVM_bow.pkl              # Trained Logistic Regression model (BoW features)
├── SVM_tfidf.pkl            # Trained Logistic Regression model (TF-IDF features)
├── count_vectorizer.pkl     # CountVectorizer object
├── tfidf_vectorizer.pkl     # TfidfVectorizer object
├── label_binarizer.pkl      # LabelBinarizer object
└── templates/
    └── index.html           # HTML template for Flask app
```

---

## Usage

### Train the Model
Run the script to preprocess the data, train models, and save them:
```bash
python train_model.py
```

### Predict Sentiment via Flask
1. Start the Flask app:
   ```bash
   python app.py
   ```
2. Input text or upload a file to get sentiment predictions.

---

## Results

- **Bag of Words Logistic Regression Accuracy:** X%
- **TF-IDF Logistic Regression Accuracy:** Y%
- **Bag of Words SVM Accuracy:** Z%
- **TF-IDF SVM Accuracy:** A%

---

## Future Enhancements
- Include more preprocessing techniques like lemmatization.
- Add advanced models such as neural networks.
- Expand the web interface for real-time sentiment analysis. 

## License
This project is open-source and available for modification and distribution.
