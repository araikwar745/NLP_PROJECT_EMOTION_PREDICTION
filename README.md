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
│   ├── BiGRU_Model.keras      # trained model (written by the notebook)
│   └── tokenizer.pkl          # tokenizer fitted on the training split
├── static/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── final_clean.ipynb          # data, training, evaluation, export
├── main.py                    # FastAPI backend
├── report.md                  # project report
├── requirements.txt
├── requirements-test.txt      # API regression test dependencies
├── render.yaml                # optional Render Blueprint
├── tests/                     # checks using the real saved model
├── .github/workflows/         # Linux checks before and after merging
├── .python-version            # Python version used by Render
└── runtime.txt
```

## Run Locally

**Prerequisites:** Python 3.12 and pip (Python 3.11 is also supported).
The pinned dependencies require Python 3.11 or 3.12.

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

## Deploying an update to the existing Render app

Merge the PR into `main`. If the existing Render service is connected to this
repository's `main` branch with auto-deploy enabled, Render builds and deploys
the merged code automatically. The trained model and tokenizer are committed
in `Artifacts/`: no notebook run, manual upload, or new service is needed.

The existing Python service can keep these standard commands:

```bash
# Build
pip install -r requirements.txt
# Start (one worker avoids loading multiple model copies)
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
```

`.python-version` selects Python 3.12. A dashboard `PYTHON_VERSION` override
using **3.11.x or 3.12.x can stay as it is**. Only an incompatible override
(for example 3.13 or 3.14 with these pinned packages) needs to be removed or
changed to `3.12.12`. Render environment variables take precedence over the file.

If the app does not update after merging, the service owner should check that
the linked branch is `main`, auto-deploy is enabled, and the build succeeded.
GitHub Actions runs on pushes to `main` as well as PRs, so Render's
"After CI Checks Pass" mode also has checks to wait for.

`render.yaml` contains the Python version, build/start commands, auto-deploy,
and `/health` check for services managed through a Render Blueprint. **Adding
this file does not change an existing dashboard-created service's settings.**
There is no need to create another service to deploy this PR.

After deployment, `/health` should return `model_loaded: true`, "I love you so
much" should return **love**, and punctuation-only input should show an error
without the previous prediction remaining visible.

See Render's [auto-deploy documentation](https://render.com/docs/deploys),
[Python version precedence](https://render.com/docs/python-version), and
[Blueprint setup](https://render.com/docs/infrastructure-as-code).

## Automated app checks

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

These checks load the committed model and tokenizer, test the API and static
assets from outside the repository directory, verify known prediction
regressions and probability output, and exercise invalid input and readiness
failures. GitHub Actions runs them on Linux with Python 3.11 and 3.12. No
Render credentials or retraining are needed.

The Python 3.12 job also opens the actual page in Chromium and checks visible
probability bars, errors clearing previous results, overlapping submissions,
and the mobile layout. To run the browser checks locally with the app running
on port 8765:

```bash
python -m pip install -r requirements-browser.txt
python -m playwright install chromium
python tests/check_browser.py http://127.0.0.1:8765
```

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

The response has `predicted_emotion`, `confidence` and `all_probabilities` (one value per emotion).
The old misspelled field `all_probabilites` is still sent so older clients keep working.

Text with no English word the model knows (for example only symbols, random letters or another
script) is rejected with HTTP 422 and a short message, instead of returning a meaningless guess.

## Retraining the Model

Run `final_clean.ipynb` from top to bottom. It trains on the `train` split, uses the `validation`
split for early stopping and model choice, reports the `test` split once, and saves
`Artifacts/BiGRU_Model.keras` and `Artifacts/tokenizer.pkl`, which `main.py` loads.

## Model Results

Measured by running `final_clean.ipynb` (seed 42, CPU). The model is chosen on the validation split; the
test split (2,000 sentences) is only used for these final numbers.

| Model | Validation accuracy | Test accuracy |
|---|---:|---:|
| Simple RNN | 83.7% | 81.0% |
| LSTM | 90.9% | 90.1% |
| GRU | 92.1% | 91.1% |
| **BiGRU (deployed)** | 92.7% | **91.7%** |

BiGRU per emotion on the test split (F1): sadness 0.957, joy 0.936, anger 0.930, fear 0.874,
love 0.831, surprise 0.739 (macro F1 0.878). Love and surprise are the rarest classes and the hardest.


