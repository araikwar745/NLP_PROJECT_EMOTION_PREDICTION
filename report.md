# NLP Project Report
## Emotion Prediction from Text Using Deep Learning

**Submitted to:**  
**Dr. Soumajit Pramanik**

**Submitted by:**  
Harshit Kumar Sahu (P25MT501)  
Praveen (12341630)  
Roshan Raj (12341830)  
M. Sukanya (B24CS028)  
Ashish Raikwar (B24CS010)

**Project Repository:** [NLP_PROJECT_EMOTION_PREDICTION](https://github.com/araikwar745/NLP_PROJECT_EMOTION_PREDICTION)  
**Live Project:** [Emotion Prediction Web Application](https://nlp-project-emotion-prediction.onrender.com/)

---

# Abstract

Human emotions are frequently expressed through written language in messages, comments, reviews, and online conversations. Identifying the emotion behind a sentence can help applications understand the tone and intent of text. This project, **Emotion Prediction from Text Using Deep Learning**, presents a web-based system that predicts the most likely emotion expressed in an English text input.

The system uses a **Bidirectional Gated Recurrent Unit (BiGRU)** neural network for emotion classification. The input sentence is cleaned and normalized, converted into a numerical sequence using a trained tokenizer, and padded to a fixed length before being passed to the model. The model predicts scores for six emotion categories: sadness, joy, love, anger, fear, and surprise. The application displays the predicted emotion and its confidence, along with the probability distribution across all six classes.

A FastAPI backend connects the trained model to a browser-based user interface. The model and tokenizer are loaded when the application starts, while API endpoints handle health checks and prediction requests. The project is deployed online so that users can try the application without setting up the development environment.

This report describes the problem, objectives, methodology, model architecture, application design, implementation, testing approach, team contributions, limitations, and possible future improvements.

**Keywords:** Natural Language Processing, Emotion Detection, Text Classification, Deep Learning, BiGRU, FastAPI, Tokenization.

---

# 1. Introduction

## 1.1 Background

Natural Language Processing (NLP) is a field of Artificial Intelligence that enables computers to process, understand, and analyze human language. A large amount of communication now takes place through text, including social media posts, online reviews, chat messages, and feedback forms.

Although computers can process words and sentences, understanding the emotion expressed by a person is more challenging. Two sentences may use similar words but communicate different feelings depending on their context. Emotion prediction attempts to identify the emotional category conveyed by a given text.

For example:

- “I am extremely happy today” expresses **joy**.
- “I miss those wonderful days” may express **sadness**.
- “I am really angry about this” expresses **anger**.
- “I am afraid of what might happen” expresses **fear**.

The objective of this project is to develop a practical system that accepts a text sentence and predicts the emotion most strongly associated with it.

## 1.2 Project Description

The project is an NLP-based emotion prediction application built using a deep learning model. It accepts an English sentence from the user and processes it through a trained BiGRU model.

The model classifies the input into one of six emotion categories:

| Emotion | Description |
|---|---|
| Sadness | Text expressing sorrow, disappointment, or unhappiness |
| Joy | Text expressing happiness, excitement, or pleasure |
| Love | Text expressing affection, care, or attachment |
| Anger | Text expressing irritation, frustration, or hostility |
| Fear | Text expressing worry, anxiety, or threat |
| Surprise | Text expressing unexpectedness or astonishment |

The result is returned through a backend API and presented through the web interface.

## 1.3 Problem Statement

Understanding emotions from text manually can be time-consuming, especially when a large number of messages need to be analyzed. Simple keyword-based approaches may also fail when the same word is used in different contexts.

Therefore, there is a need for an automated text classification system that can learn patterns from language data and predict the likely emotion expressed in a sentence.

This project addresses the problem by using a trained deep learning model and providing an easy-to-use web application for emotion prediction.

## 1.4 Objectives

The main objectives are:

1. To study the application of NLP in emotion recognition.
2. To preprocess raw text into a consistent format.
3. To convert text into numerical sequences using tokenization.
4. To use a BiGRU deep learning model for emotion classification.
5. To predict one of six emotion categories.
6. To provide confidence and class-wise probability scores.
7. To develop a backend API for model inference.
8. To connect the prediction service with a user-friendly web interface.
9. To deploy the application online for easy access.

---

# 2. Literature and Technical Background

## 2.1 Natural Language Processing

Natural Language Processing combines concepts from computer science, linguistics, and machine learning. It allows machines to work with human language in tasks such as sentiment analysis, text classification, translation, summarization, and question answering.

In this project, NLP is used to prepare text data and represent sentences in a format that a neural network can process.

## 2.2 Text Classification

Text classification is the process of assigning a predefined category to a piece of text. It is commonly used in spam detection, topic classification, sentiment analysis, and emotion recognition.

Emotion prediction is a multi-class classification task because the model chooses among six possible emotion labels.

## 2.3 Recurrent Neural Networks

A Recurrent Neural Network (RNN) is a neural network architecture designed to process sequential data. It maintains information from earlier elements in a sequence, which makes it useful for language-related tasks.

However, basic RNNs can face difficulties learning long-term dependencies. Gated architectures such as LSTM and GRU help address this issue.

## 2.4 Gated Recurrent Unit (GRU)

A GRU is a type of recurrent neural network that uses gates to control how information is retained or updated. It has two main gates:

- **Update gate:** Controls how much previous information should be retained.
- **Reset gate:** Controls how much previous information should be forgotten while processing the current input.

GRUs provide a comparatively compact alternative to LSTM networks while still capturing sequential information.

## 2.5 Bidirectional GRU (BiGRU)

A Bidirectional GRU processes a sequence in two directions:

- Forward direction: reads the sequence from beginning to end.
- Backward direction: reads the sequence from end to beginning.

The two directions allow the network to use information from both preceding and following words when learning sentence representations. This can be useful for emotion classification because the meaning of a word may depend on the surrounding context.

The trained BiGRU model used in this project is stored in the repository as `Artifacts/BiGRU_Model.keras`.

---

# 3. Proposed System and Methodology

## 3.1 System Overview

The proposed application follows a sequence of processing stages:

1. The user enters a sentence through the web interface.
2. The frontend sends the sentence to the backend.
3. The backend validates the input.
4. The text is cleaned using a preprocessing function.
5. The tokenizer converts words into integer sequences.
6. The sequence is padded or truncated to the required length.
7. The BiGRU model predicts probabilities for the six emotions.
8. The emotion with the highest predicted probability is selected.
9. The backend returns the predicted emotion, confidence, and all class probabilities.
10. The frontend displays the prediction to the user.

## 3.2 System Architecture

```text
              USER
                |
                v
       Web User Interface
                |
                v
       FastAPI Backend
                |
                v
        Input Validation
                |
                v
        Text Preprocessing
                |
                v
       Trained Tokenizer
                |
                v
       Sequence Padding
                |
                v
        BiGRU Deep Learning
              Model
                |
                v
       Emotion Probabilities
                |
                v
       Highest Probability
                |
                v
       Prediction API Response
                |
                v
       Result on Web Interface
```

## 3.3 Emotion Categories

The system uses six output labels in the following order:

1. Sadness
2. Joy
3. Love
4. Anger
5. Fear
6. Surprise

The predicted class is selected by finding the index of the highest output probability. An emoji is also mapped to each emotion to make the result easier to recognize visually.

## 3.4 Text Preprocessing

Raw text may contain uppercase letters, punctuation, apostrophes, and unnecessary spaces. The preprocessing function makes the input more consistent before tokenization.

The implementation performs these operations:

- Converts all text to lowercase.
- Removes apostrophes.
- Replaces characters other than English letters, digits, and whitespace with spaces.
- Replaces repeated whitespace with a single space.
- Removes leading and trailing spaces.

For example:

**Original input:**  
`"I can't believe this happened!!!"`

**After preprocessing:**  
`i cant believe this happened`

Consistent preprocessing is important because the model should receive text in a format similar to the one used during training.

## 3.5 Tokenization

A neural network cannot directly process ordinary words. The tokenizer converts words into integer identifiers based on the vocabulary learned during training.

The trained tokenizer is stored in:

`Artifacts/tokenizer.pkl`

During prediction, the tokenizer converts the cleaned sentence into a sequence of integers. The same tokenizer must be used during training and inference so that word identifiers remain consistent.

## 3.6 Sequence Padding and Truncation

Different sentences contain different numbers of words. Neural network input batches generally require sequences of a consistent length.

This project uses a maximum sequence length of **50**. The backend applies post-padding and post-truncation:

- If a sequence has fewer than 50 tokens, padding is added at the end.
- If a sequence has more than 50 tokens, extra tokens at the end are removed.

This produces a uniform input shape for the model. The padding zeros are masked (`mask_zero=True` in the Embedding layer), so the recurrent layers skip them and only read the real words. Without the mask, a short sentence is followed by up to 49 zeros and the model partly forgets it.

---

# 4. Model Development and Prediction

## 4.1 BiGRU Model

The central prediction component is a trained Bidirectional GRU model. It learns sequential patterns in text and uses the learned representation to classify the sentence into one of the six emotion categories.

The trained model is loaded from:

`Artifacts/BiGRU_Model.keras`

The model is loaded when the FastAPI application starts. This avoids loading the model separately for every prediction request.

## 4.2 Prediction Process

Let the model output a probability vector:

\[
P = [p_1,p_2,p_3,p_4,p_5,p_6]
\]

where each value represents the predicted probability associated with one emotion class.

The final emotion is selected using:

\[
\text{Predicted Class} = \arg\max_i(p_i)
\]

The confidence score is the probability corresponding to the selected class.

For example, the following values are only an illustration of the response format, not a measured result from the trained model:

| Emotion | Illustrative probability |
|---|---:|
| Sadness | 0.05 |
| Joy | 0.78 |
| Love | 0.08 |
| Anger | 0.02 |
| Fear | 0.03 |
| Surprise | 0.04 |

In this illustrative case, the predicted emotion would be **Joy**, with a confidence score of 0.78.

## 4.3 Model Output

The backend returns the following information:

- Original input text
- Predicted emotion
- Confidence score
- Probability for each emotion category

Returning the complete probability distribution makes the result more informative than returning only a single label.

## 4.4 Model and Tokenizer Management

The application uses a FastAPI lifespan function to load the BiGRU model and tokenizer during startup. They are stored in application memory and reused for incoming requests.

When the application shuts down, the stored model references are cleared. This approach supports efficient inference and avoids repeatedly loading the model from disk.

---

# 5. Application Design and Implementation

## 5.1 Technology Stack

The project combines deep learning, backend development, and web interface technologies.

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | Loading and running the trained neural network |
| BiGRU | Emotion classification model |
| NumPy | Numerical operations on prediction outputs |
| Pickle | Loading the saved tokenizer |
| FastAPI | Backend API development |
| Pydantic | Request and response data validation |
| HTML, CSS and JavaScript | Browser-based user interface |
| Git and GitHub | Source code management |
| Render | Online deployment |

The exact frontend implementation can be inspected in the repository's `static` directory.

## 5.2 Backend Development with FastAPI

FastAPI is used to expose the trained model through HTTP endpoints. It provides request validation and structured responses.

The main backend file is:

`main.py`

The application defines a text input schema with a minimum length of one character and a maximum length of 2,000 characters.

### Main API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Serves the web application homepage |
| `/health` | GET | Checks whether the server and model are available |
| `/predict` | POST | Accepts text and returns an emotion prediction |

### Health Check

The `/health` endpoint returns the server status and whether the model has been loaded. This helps check the availability of the prediction service.

### Emotion Prediction

The `/predict` endpoint accepts a JSON request containing a text field. It validates the input, preprocesses the sentence, tokenizes it, pads the sequence, and passes it to the BiGRU model.

If the model or tokenizer is not available, the backend returns an HTTP 503 error rather than attempting an invalid prediction. If the cleaned text is empty, or none of its words are in the model's vocabulary, the backend returns HTTP 422 with a short message instead of a meaningless guess.

## 5.3 API Response Structure

A successful prediction response contains:

```json
{
  "text": "I feel so happy and excited",
  "predicted_emotion": "joy",
  "confidence": 0.0,
  "all_probabilities": {
    "sadness": 0.0,
    "joy": 0.0,
    "love": 0.0,
    "anger": 0.0,
    "fear": 0.0,
    "surprise": 0.0
  },
  "all_probabilites": { "...": "same values as all_probabilities" }
}
```

The numeric values above are placeholders to explain the response structure. Actual values are generated by the model for each input. The field `all_probabilites` is the old misspelled name; it is still sent with the same values so that older clients keep working.

## 5.4 Frontend and User Interaction

The frontend provides the interface through which users interact with the prediction system. It is served by the FastAPI application from the `static` directory.

The frontend and backend work together to make the application usable without requiring users to interact directly with Python code or API requests.

The user-facing workflow is:

1. Open the deployed application.
2. Enter a sentence in the text input area.
3. Submit the sentence for analysis.
4. Wait for the prediction response.
5. View the predicted emotion and its confidence/probability information.

## 5.5 Deployment

The project is hosted on Render and can be accessed through the live deployment link:

[https://nlp-project-emotion-prediction.onrender.com/](https://nlp-project-emotion-prediction.onrender.com/)

The source code and supporting files are maintained in GitHub:

[https://github.com/araikwar745/NLP_PROJECT_EMOTION_PREDICTION](https://github.com/araikwar745/NLP_PROJECT_EMOTION_PREDICTION)

The repository includes the backend application, model artifacts, static frontend files, dependency information, runtime configuration, and the model development notebook.

---

# 6. Testing and Results

## 6.1 Testing Objectives

Testing is important to check whether the application accepts valid input, processes it correctly, and returns a structured response.

The main testing objectives are:

- Verify that the application starts and loads its model artifacts.
- Check whether the homepage is accessible.
- Verify that the health endpoint reports model availability.
- Check that valid text is accepted by the prediction endpoint.
- Confirm that the response contains a predicted emotion and probability values.
- Check that invalid or unavailable-model situations are handled appropriately.

## 6.2 Functional Test Cases

| Test Case | Input / Action | Expected Behaviour |
|---|---|---|
| Homepage | Open `/` | Web interface is served |
| Health check | Send GET request to `/health` | Server and model status are returned |
| Valid text | Submit a non-empty sentence | Prediction response is returned |
| Empty text | Submit an empty string | Input validation rejects the request |
| Long input | Submit text exceeding 2,000 characters | Input validation rejects the request |
| No known words | Submit only symbols, random letters or non-English script | HTTP 422 with a clear message |
| Model unavailable | Request prediction when artifacts are not loaded | Service returns an unavailable-model error |
| Probability output | Submit valid text | Six emotion probability values are returned |

These cases describe the intended functional checks. They should not be interpreted as a record of automated test execution unless supported by actual test logs.

## 6.3 Result Interpretation

For each valid input, the model produces a score for each of the six emotion categories. The category with the highest score is selected as the final prediction.

The confidence value helps communicate how strongly the model favors its selected category relative to its other output scores. It should not automatically be interpreted as a guarantee that the prediction is correct.

The application also returns all class-wise scores, which can help users understand cases where more than one emotion receives a relatively high score.

## 6.4 Measured Model Results

The models were trained on the `train` split (16,000 sentences) of `dair-ai/emotion`. The `validation` split (2,000) was used for early stopping and for choosing the model. The `test` split (2,000) was used only for the final numbers below. All padding is masked (`mask_zero=True`).

| Model | Validation accuracy | Test accuracy |
|---|---:|---:|
| Simple RNN | 83.7% | 81.0% |
| LSTM | 90.9% | 90.1% |
| GRU | 92.1% | 91.1% |
| **BiGRU (deployed)** | 92.7% | **91.7%** |

Per-class results of the BiGRU on the test split:

| Emotion | Precision | Recall | F1-score | Sentences |
|---|---:|---:|---:|---:|
| Sadness | 0.975 | 0.940 | 0.957 | 581 |
| Joy | 0.962 | 0.912 | 0.936 | 695 |
| Love | 0.763 | 0.912 | 0.831 | 159 |
| Anger | 0.915 | 0.945 | 0.930 | 275 |
| Fear | 0.934 | 0.821 | 0.874 | 224 |
| Surprise | 0.591 | 0.985 | 0.739 | 66 |
| **Macro average** | 0.857 | 0.919 | 0.878 | 2000 |

Love and surprise are the rarest classes. Because of the balanced class weights, the model predicts them often (high recall) but is less precise on them. Earlier versions validated on the test split and did not mask the padding; in that setup the plain RNN, LSTM and GRU stayed near chance (loss ≈ ln 6 ≈ 1.79), and short sentences such as "I love you so much" were misclassified.



---

# 7. Team Contributions

The project was developed collaboratively. The machine learning and core application development work was carried out by Ashish, Harshit, and Sukanya, while Praveen and Roshan focused on the user interface.

The following contribution breakdown presents the team's work in a clear and organized manner.

| Team Member | Roll Number | Contribution |
|---|---|---|
| **Ashish Raikwar** | B24CS010 | Core project development, NLP pipeline, BiGRU model integration, backend/API implementation, prediction workflow, and deployment integration |
| **Harshit Kumar Sahu** | P25MT501 | Support in project development, data and model-related work, reviewing the prediction pipeline, and overall project coordination |
| **M. Sukanya** | B24CS028 | Support in NLP/model development, analysis of emotion classification, testing support, and report/documentation work |
| **Praveen** | 12341630 | User interface design and frontend implementation |
| **Roshan Raj** | 12341830 | User interface development, frontend integration, and interface-related improvements |

### 7.1 Individual Role Summary

**Ashish Raikwar — Core Development and Integration**

Worked on the main project implementation, including the NLP processing pipeline, integration of the trained BiGRU model, backend prediction logic, API functionality, and deployment-related integration. Also contributed to connecting the model output with the web application.

**Harshit Kumar Sahu — Model and Project Development Support**

Contributed to the machine learning and project development work, including support for the data/model workflow, reviewing the prediction process, and coordination of the overall implementation.

**M. Sukanya — NLP and Evaluation Support**

Contributed to the NLP and model-related work, supported analysis of the emotion classification process, and assisted with testing and technical documentation.

**Praveen — UI Development**

Focused on the user interface, including the frontend structure and presentation of the application to make interaction straightforward.

**Roshan Raj — UI Development and Integration**

Worked on frontend development and interface-related integration to connect the user-facing application with the prediction workflow.

---

# 8. Advantages, Applications and Limitations

## 8.1 Advantages

1. **Automated prediction:** Reduces the need to manually categorize every sentence.
2. **Context-aware sequence processing:** BiGRU can use information from both directions of a text sequence.
3. **Multiple emotion categories:** Predicts six different emotions instead of only positive or negative sentiment.
4. **Probability output:** Provides scores for all emotion classes.
5. **Web-based access:** Users can interact with the system through a browser.
6. **Reusable backend:** FastAPI exposes the model through a structured API.
7. **Online availability:** The deployed application can be accessed without local installation.

## 8.2 Possible Applications

The system demonstrates a basic approach that could be useful in several areas:

- **Social media analysis:** Studying emotional patterns in public text.
- **Customer feedback:** Identifying emotions expressed in written feedback.
- **Text analytics:** Adding emotion labels to collections of text.
- **Chat applications:** Exploring emotion-aware features in text-based interfaces.
- **Academic research:** Demonstrating deep learning techniques for NLP classification.

These are potential applications; the project is an educational prototype and should not be treated as a validated system for high-stakes decisions.

## 8.3 Limitations

The current implementation has some limitations:

1. It uses six predefined emotion categories and cannot represent every possible human emotion.
2. The current preprocessing function is designed for English text and removes many non-English characters.
3. Sarcasm, irony, slang, and complex context may be difficult to interpret correctly.
4. The model's prediction depends on the quality and coverage of its training data.
5. A confidence score does not guarantee that a prediction is correct.
6. The fixed sequence length of 50 tokens may remove useful information from longer sentences.
7. Emotion is subjective, and a sentence may express more than one emotion at the same time.

---

# 9. Future Scope and Conclusion

## 9.1 Future Enhancements

The project can be improved in several ways:

- **Multilingual support:** Extend preprocessing and model training to support Hindi and other languages.
- **Hinglish support:** Explore mixed Hindi-English text processing for commonly used informal messages.
- **More emotion categories:** Add additional labels such as disgust, gratitude, confusion, and frustration.
- **Improved contextual understanding:** Experiment with transformer-based models such as BERT or other suitable language models.
- **Better handling of long text:** Explore more flexible sequence lengths or document-level processing.
- **Explainability:** Provide useful information about which parts of a sentence influence a prediction.
- **Performance improvements:** Optimize inference time and resource usage for deployment.
- **User feedback:** Allow users to report incorrect predictions for future model improvement.

## 9.2 Conclusion

The **Emotion Prediction from Text Using Deep Learning** project demonstrates how Natural Language Processing and deep learning can be combined to identify emotions expressed in written text.

A BiGRU model is used to classify text into six emotion categories: sadness, joy, love, anger, fear, and surprise. The input is cleaned, tokenized, and converted into a fixed-length numerical sequence before being processed by the trained model. The backend, developed using FastAPI, provides prediction and health-check endpoints, while the web interface allows users to interact with the system conveniently.

The project provided practical experience in text preprocessing, neural network inference, API development, frontend-backend integration, version control, and cloud deployment. It also demonstrates how a trained NLP model can be converted into an accessible web application.

Overall, the project establishes a working foundation for text-based emotion classification and offers several opportunities for further improvement, especially in multilingual processing, contextual understanding, and model evaluation.

---

# References

1. **TensorFlow Documentation:** https://www.tensorflow.org/
2. **Keras Documentation:** https://keras.io/
3. **FastAPI Documentation:** https://fastapi.tiangolo.com/
4. **NumPy Documentation:** https://numpy.org/doc/
5. **Python Documentation:** https://docs.python.org/3/
6. **GitHub Repository — Project Source Code:** https://github.com/araikwar745/NLP_PROJECT_EMOTION_PREDICTION
7. **Live Project Deployment:** https://nlp-project-emotion-prediction.onrender.com/

---

## End of Report
