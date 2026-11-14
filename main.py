
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from keras.models import load_model
from pathlib import Path
import numpy as np
import pickle
import os
import re




"""
1. We are going to make some constants like:
A. Model Path (BiGRU)
B. Tokenizer Path
C. Max Sequence Length
D. Emotion Labels
"""
#A. Model Path (BiGRU) - can be changed with the MODEL_PATH environment variable
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / os.getenv("MODEL_PATH", "Artifacts/BiGRU_Model.keras")

#B. Tokenizer Path - can be changed with the TOKENIZER_PATH environment variable
tokenizer_path = BASE_DIR / os.getenv("TOKENIZER_PATH", "Artifacts/tokenizer.pkl")

#C. Max Sequence Length (must match the notebook)
max_sequence_length = 50

#D. Emotion Labels (same order as the dair-ai/emotion dataset)
emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]



"""
2. Preprocess the upcoming text
Cleans raw text so it matches the format used while training (the notebook uses the same function).
A. Convert the text to lowercase. -done
B. Remove apostrophes (e.g can't -> cant). -done
C. Remove Special Characters and Punctuation. -done
D. Remove extra spaces -done
"""

def preprocess_text(text: str)->str:
    text = text.lower()
    text = re.sub(r"'","",text)
    text = re.sub(r"[^a-z0-9\s]"," ", text)
    text = re.sub(r"\s+", " ",text).strip()
    return text


"""
3. Request and Response Schemas
A. Text Input -> Input schema the text sent by user. -done
B. Prediciton Response -> Output schema the emotion to predict. -done
C. Health Response (Server health check)
"""

class TextInput(BaseModel):
    text : str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={"example": "I feel so happy and excited"}
        )

class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence : float
    all_probabilities: dict[str, float]
    # old misspelled name, kept so existing clients do not break
    all_probabilites: dict[str, float]

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # allows the field name "model_loaded" without a warning

    status: str
    model_loaded: bool

"""
4. Model Loading and LifeSpan Management
Load the model and toknizer once the server starts up.
"""
dl_model = {} #{1. BiGRU, 2. Tokenizer}-> True , {} -> False

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Loading the model and tokenizer...')
    # Serving only needs the weights, not the training optimizer and metrics.
    dl_model["BiGRU"] = load_model(model_path, compile=False)
    with open(tokenizer_path, 'rb') as file:
        dl_model["Tokenizer"] = pickle.load(file)
    print('Model are loaded successfully...')

    yield #Pause, model is laoded and server is running and at this point model wait karega for request

    dl_model.clear() #Ek baar server band ho gaya uske baad model ko memory se hata do.


"""
5. Mount the static files to the FastAPI app
The web page is served by this same app, so no CORS middleware is needed.
"""
app = FastAPI(
    lifespan=lifespan
)

app.mount('/static', StaticFiles(directory=BASE_DIR / "static"), name="static")



"""
6. API Endpoints.
A. Server UI at homepage ('/')
B. Health Check Endpoint ('/health')
C. Predict Emotion Endpoint ('/predict')
"""

#A. Server UI at homepage ('/')
@app.get('/', include_in_schema=False)
def server_ui():
    return FileResponse(BASE_DIR / 'static/index.html')

#B. Health Check Endpoint ('/health')
@app.get('/health', response_model=HealthResponse)
def health_check():
    ready = all(dl_model.get(key) is not None for key in ("BiGRU", "Tokenizer"))
    if not ready:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return HealthResponse(status="Server is running", model_loaded=True)

#C. Predict Emotion Endpoint ('/predict')
@app.post('/predict', response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):
    """
    1. Cleans the input sentences.
    2. Convert the words into numeric using tokenizer.
    3. Reject input with no English word the model knows (it would only get a meaningless guess).
    4. Pad the sequences to ensure uniform length.
    5. Run prediction using the BiGRU model.
    6. Return the top emotion and full probability breakdown.
    """

    BiGRU_model     = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("Tokenizer")

    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again later.")

    #1.
    cleaned_text = preprocess_text(text_input.text)
    if not cleaned_text:
        raise HTTPException(status_code=422, detail="Please type a sentence in English. No English letters or numbers were found in the text.")

    #2.
    tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])
    # Validate the same first 50 tokens that will actually reach the model.
    tokenized_text = [tokenized_text[0][:max_sequence_length]]

    #3. index 1 is the <unk> token for words outside the vocabulary
    oov_index = tokenizer_model.word_index.get(tokenizer_model.oov_token)
    if not any(token != oov_index for token in tokenized_text[0]):
        raise HTTPException(status_code=422, detail="None of these words are in the model's English vocabulary. Please try a normal English sentence.")

    #4.
    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post"
    )

    #5. a direct call is faster than model.predict() for a single sentence
    probabilities = np.asarray(BiGRU_model(padded_sequence, training=False))[0]

    top_emotion_index = int(np.argmax(probabilities))
    all_probabilities = {
        label: float(prob) for prob, label in zip(probabilities, emotion_labels)
    }

    return PredictionResponse(
        text = text_input.text,
        predicted_emotion = emotion_labels[top_emotion_index],
        confidence = float(probabilities[top_emotion_index]),
        all_probabilities = all_probabilities,
        all_probabilites = all_probabilities,
    )
