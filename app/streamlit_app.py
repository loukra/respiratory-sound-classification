# Respiratory Health Classifier — Streamlit app

import os
import sys
import urllib.request

import numpy as np
import soundfile as sf
import librosa
import streamlit as st
from PIL import Image

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

import predict  # noqa: E402
import preprocess  # noqa: E402

MODEL_URL = (
    "https://github.com/loukra/respiratory-sound-classification"
    "/releases/download/v1.0.0/ResNet.h5"
)
MODEL_PATH = os.path.join(APP_DIR, os.pardir, "models", "ResNet.h5")

TARGET_SR = 4000
MIN_SECONDS = 16

st.set_page_config(
    page_title="Respiratory Health Classifier",
    page_icon="🫁",
    layout="centered",
)

with open(os.path.join(APP_DIR, "style.css")) as f:
    design = f.read()
st.html(f"""
<style>
{design}
</style>
<div class="hero">
  <h1>You Breathe, We Classify</h1>
  <p>AI-assisted screening from 16 seconds of lung sounds</p>
  <span class="dot"></span>
</div>
""")


def ensure_model_file():
    """Download the model from the GitHub release with a visible progress bar."""
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    bar = st.progress(0.0, text="Downloading the neural network (270 MB)...")

    def hook(blocks, block_size, total_size):
        if total_size > 0:
            pct = min(blocks * block_size / total_size, 1.0)
            bar.progress(pct, text=f"Downloading the neural network... {int(pct * 100)} %")

    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=hook)
    bar.empty()


@st.cache_resource(show_spinner="Initializing the neural network...")
def load_model():
    import keras
    model = keras.models.load_model(MODEL_PATH, compile=False)
    # dummy inference so TF traces the graph now, not on the first user request
    model.predict(np.zeros((1, 224, 224, 3)), verbose=0)
    return model


def result_card(y_pred):
    healthy = y_pred > 0.5
    pct = 5 * round((y_pred if healthy else 1 - y_pred) * 100 / 5)
    cls = "healthy" if healthy else "diseased"
    text = (
        "probability that your respiratory system is <b>healthy</b>"
        if healthy
        else "probability that your respiratory system is <b>diseased</b> — "
             "consider seeking a doctor's opinion"
    )
    st.html(f"""
<div class="result-card {cls}">
  <div class="result-pct">≈ {pct} %</div>
  <div class="result-text">{text}</div>
  <div class="gauge"><div class="gauge-fill" style="width:{pct}%"></div></div>
  <div class="result-note">This is an AI estimate from a student research project,
  not a medical diagnosis. If in doubt, always consult a doctor.</div>
</div>
""")


with st.sidebar:
    st.title("Respiratory Health Classifier")
    st.markdown("""This project is designed to detect whether your respiratory system is diseased or healthy. To get a reasonable result:
  * Take off your shirt.
  * Place your microphone directly on your skin in the position shown in the picture.
  * Start the recording by pressing the microphone button.
  * Breathe deeply in and out through your mouth for at least 16 seconds.
  * Stop the recording — the analysis starts automatically.

  The evaluation of the recording is performed by an artificial neural network. The result is displayed a few seconds after the recording.
  """)
    st.image(Image.open(os.path.join(APP_DIR, "adam.jpg")))
    st.markdown("[Picture Taken from A.D.A.M](https://ssl.adam.com/graphics/images/en/23267.jpg)")


ensure_model_file()
model = load_model()

audio = st.audio_input(f"Record your breathing for at least {MIN_SECONDS} seconds")

if audio is not None:
    data_origin, samplerate = sf.read(audio)

    # recordings may be mono or stereo depending on the device
    wav = data_origin[:, 0] if data_origin.ndim > 1 else data_origin

    data = librosa.resample(y=wav, orig_sr=samplerate, target_sr=TARGET_SR)
    duration = data.shape[0] / TARGET_SR

    if duration >= MIN_SECONDS:
        with st.spinner('Asking the Doc...'):
            preprocessor = preprocess.AudioPreprocessor()
            predictor = predict.MyPredictor(model, preprocessor)

            # trim to a whole number of seconds
            data = data[: int(TARGET_SR * np.floor(duration))]
            y_pred = predictor.predict(data)

        result_card(y_pred)
    else:
        st.error(
            f"The recording is only {duration:.0f} seconds long — "
            f"it must be at least {MIN_SECONDS} seconds to obtain a result."
        )

st.html("""
<div class="app-footer">
  Built with TensorFlow &amp; Streamlit ·
  <a href="https://github.com/loukra/respiratory-sound-classification">Model &amp; code on GitHub</a>
</div>
""")
