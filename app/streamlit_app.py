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

with open(os.path.join(APP_DIR, "style.css")) as f:
    design = f.read()
st.html(f"""
<style>
{design}
</style>
<div style="text-align:center">
  <h1 style="font-family:Arial">You Breath, We Classify</h1>
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
    st.markdown("""Please remember that this is not a medical diagnosis.
  If in doubt, it is best to seek a doctor's opinion. """)


ensure_model_file()
model = load_model()

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
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

            if y_pred > 0.5:
                st.success(
                    f"There is a higher probability of about {5 * round((y_pred * 100) / 5)} % "
                    "that your respiratory system is healthy"
                )
            else:
                st.error(
                    f"There is a higher probability of about {5 * round(((1 - y_pred) * 100) / 5)} % "
                    "that your respiratory system is diseased. "
                    "Please remember that this is not a medical diagnosis. "
                    "If in doubt, it is best to seek a doctor's opinion."
                )
        else:
            st.error(
                f"The recording is only {duration:.0f} seconds long — "
                f"it must be at least {MIN_SECONDS} seconds to obtain a result."
            )
