# Email/SMS Spam Classifier

A machine learning web application that utilizes Natural Language Processing (NLP) to classify text messages and emails as either "Spam" or "Not Spam". Built with Python and Streamlit, this project leverages a TF-IDF vectorizer and a Multinomial Naive Bayes classification model to provide real-time, highly accurate predictions.

## Features

* **Real-Time Inference:** Instantly evaluate conversational text or promotional broadcasts through an uncluttered, responsive web interface.  
* **Optimized NLP Pipeline:** Text preprocessing is heavily optimized utilizing O(1) memory lookups for NLTK stopword filtering, drastically reducing inference and training times.  
* **High-Contrast UI:** Employs intuitive visual feedback with dynamic color-coded verdict banners (Green for benign, Red for high-risk spam tokens).  
* **Cloud-Ready Configuration:** Includes robust setup.sh and Procfile configurations tailored for seamless deployment on modern Platform-as-a-Service (PaaS) environments.  
* **Modern ML Architecture:** Serialized using contemporary scikit-learn architectures to prevent deprecation and unpickling errors.

Screenshots

| Interface State | Visualization |
| :---- | :---- |
| **Initial Deployment:** The static landing page featuring the input widget and uninitialized prediction controls. | **Spam Class Word Cloud:** Dominant TF-IDF tokens ("FREE", "URGENT", "WIN") scaling dynamically based on class frequency. |
| **Positive Detection:** Emphatic red alert banner isolating high-weight promotional keywords. | **Negative Detection:** Green success banner verifying standard conversational prose. |

## Tech Stack

* **Frontend/Framework:** [Streamlit](https://streamlit.io/)  
* **Machine Learning:** [Scikit-learn](https://scikit-learn.org/) (MultinomialNB, TfidfVectorizer)  
* **Natural Language Processing:** [NLTK](https://www.nltk.org/) (PorterStemmer, Tokenization)  
* **Data Manipulation:** Pandas, NumPy  
* **Visualization (EDA):** Matplotlib, WordCloud

## Dataset & Model Evaluation

* **Dataset:** This model is trained on the [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection), consisting of 5,574 English, real, and non-encoded messages tagged according to being legitimate (ham) or spam.  
* **Algorithm:** Multinomial Naive Bayes (chosen for its exceptional performance and low false-positive rate on discrete text frequency data).  
* **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency) restricted to a maximum of 3,000 features to prevent overfitting and optimize memory usage.

## Project Structure

* 

```
├── app.py                 # Main Streamlit application script
├── train.py               # ML training and vectorization script
├── vectorizer.pkl         # Serialized TF-IDF vectorizer
├── model.pkl              # Serialized Multinomial Naive Bayes model
├── requirements.txt       # Python package dependencies
├── nltk.txt               # NLTK corpora requirements for cloud deployment
├── setup.sh               # Headless server configuration script
└── Procfile               # Cloud deployment command definitions
```

## Local Installation

1. **Clone the repository:**  
* 

```sh
git clone https://github.com/your-username/sms-spam-classifier.git
cd sms-spam-classifier
```

2. **Install dependencies:**  
* 

```sh
pip install -r requirements.txt
```

3. **Download required NLTK data:**  
* 

```sh
python -m nltk.downloader punkt_tab stopwords
```

4. **Run the application:**  
* 

```sh
streamlit run app.py
```

The application will boot and be accessible at [http://localhost:8501](http://localhost:8501).

## Retraining the Model

If you wish to update the model with a new dataset or modify the feature engineering pipeline:  
Ensure the raw dataset (spam.csv) is located in the root directory.  
Run the optimized training script:

* 

```sh
python train.py
```

The script will preprocess the dataset, fit the new vectorizer, train the Naive Bayes model, and automatically overwrite the existing vectorizer.pkl and model.pkl files.

## Deployment

This application is configured for immediate PaaS deployment (e.g., Heroku, Render). The included setup.sh dynamically builds the config.toml required to run Streamlit in a headless server environment, while the Procfile dictates the boot sequence. No manual configuration of CORS or port binding is required.

## Future Scope

* **API Integration:** Transition the backend to FastAPI to serve predictions to external mobile or web clients.  
* **Deep Learning Expansion:** Implement LSTM or BERT models to capture sequential context and semantic meaning beyond TF-IDF vectorization.  
* **User Authentication:** Add login functionality to allow users to maintain a history of their analyzed messages.

## Contributing

Contributions, issues, and feature requests are welcome\!

5. Fork the Project  
6. Create your Feature Branch (git checkout \-b feature/AmazingFeature)  
7. Commit your Changes (git commit \-m 'Add some AmazingFeature')  
8. Push to the Branch (git push origin feature/AmazingFeature)  
9. Open a Pull Request

## Acknowledgments

* Original project structure and inspiration provided via the CampusX official curriculum.

