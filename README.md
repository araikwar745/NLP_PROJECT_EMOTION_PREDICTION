# Emotion-Prediction Based Sentiment Analysis

An NLP-based deep learning application that predicts emotions from text
in real time and displays confidence scores.

## Live Demo

[Try the project](https://nlp-project-emotion-prediction.onrender.com/)

## Features

-   Predicts six emotions: **Sadness, Joy, Love, Anger, Fear, and
    Surprise**
-   Displays the predicted emotion and its confidence score
-   Provides a probability breakdown for all six emotions
-   Includes a simple web interface and a FastAPI backend

## Tech Stack

-   **Language:** Python
-   **Model:** Bidirectional GRU (BiGRU)
-   **Deep Learning:** TensorFlow / Keras
-   **Backend:** FastAPI
-   **Other:** NumPy, Pydantic, HTML, CSS, JavaScript

## How It Works

1.  Enter a sentence in the web interface.
2.  The input text is cleaned and preprocessed.
3.  The tokenizer converts the text into a sequence and pads it to the
    required length.
4.  The trained BiGRU model predicts the emotion.
5.  The application displays the predicted emotion, confidence score,
    and probabilities for all classes.

## Project Structure

``` text
NLP_PROJECT_EMOTION_PREDICTION/
├── Artifacts/
│   ├── BiGRU_Model.keras
│   └── tokenizer.pkl
├── static/
│   └── index.html
├── main.py
├── requirements.txt
└── runtime.txt
```

## Run Locally

**Prerequisites:** Python 3.11.9 and pip.

1.  Clone the repository:

    ``` bash
    git clone https://github.com/araikwar745/NLP_PROJECT_EMOTION_PREDICTION.git
    cd NLP_PROJECT_EMOTION_PREDICTION
    ```

2.  Create and activate a virtual environment:

    ``` bash
    python -m venv venv
    ```

    Windows:

    ``` bash
    venv\Scripts\activate
    ```

    macOS / Linux:

    ``` bash
    source venv/bin/activate
    ```

3.  Install dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

4.  Start the FastAPI server:

    ``` bash
    uvicorn main:app --reload
    ```

5.  Open **http://127.0.0.1:8000** in your browser.

Ensure the trained model and tokenizer files are present in the
`Artifacts/` directory before starting the server.

## API Endpoints

  Method   Endpoint     Description
  -------- ------------ --------------------------------------
  `GET`    `/`          Opens the web interface
  `GET`    `/health`    Checks server and model status
  `POST`   `/predict`   Predicts the emotion from input text

Example request body for `/predict`:

``` json
{
  "text": "I feel so happy and excited"
}
```


