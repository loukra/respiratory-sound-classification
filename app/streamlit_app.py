# Respiratory Health Classifier — Streamlit app
# Audio recorder component based on streamlit_audio_recorder by stefanrmmr (April 2022)

import io
import os
import sys
import urllib.request

import numpy as np
import soundfile as sf
import librosa
import streamlit as st
import streamlit.components.v1 as com
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

# Design tweaks to the standard streamlit UI/UX
st.markdown('''<style>.css-1egvi7u {margin-top: -3rem;}</style>''',
            unsafe_allow_html=True)
st.markdown('''<style>.stAudio {height: 45px;}</style>''',
            unsafe_allow_html=True)

# Custom REACT-based component for recording client audio in the browser
build_dir = os.path.join(APP_DIR, "st_audiorec", "frontend", "build")
st_audiorec = com.declare_component("st_audiorec", path=build_dir)

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


@st.cache_resource(show_spinner="Loading the neural network...")
def load_model():
    import keras
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    model = keras.models.load_model(MODEL_PATH, compile=False)
    # dummy inference so TF traces the graph now, not on the first user request
    model.predict(np.zeros((1, 224, 224, 3)), verbose=0)
    return model


with st.sidebar:
    st.title("Respiratory Health Classifier")
    st.markdown("""This project is designed to detect whether your respiratory system is diseased or healthy. To get a reasonable result:
  * Take off your shirt.
  * Place your microphone directly on your skin in the position shown in the picture.
  * Start the recording by pressing the record button.
  * Breathe deeply in and out through your mouth for at least 16 seconds.
  * Recording stops automatically after 16 seconds.

  The evaluation of the recording is performed by an artificial neural network. The result is displayed within a few seconds after the recording.
  """)
    st.image(Image.open(os.path.join(APP_DIR, "adam.jpg")))
    st.markdown("[Picture Taken from A.D.A.M](https://ssl.adam.com/graphics/images/en/23267.jpg)")
    st.markdown("""Please remember that this is not a medical diagnosis.
  If in doubt, it is best to seek a doctor's opinion. """)


# warm up the model while the user reads the instructions / records,
# so the first prediction doesn't stall on the 270 MB download
model = load_model()

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    holder = st.empty()
    with holder:
        val = st_audiorec()

    if isinstance(val, dict):
        # reassemble the component's {position: byte} dict into raw audio bytes
        ind, vals = zip(*val['arr'].items())
        ind = np.array(ind, dtype=int)
        vals = np.array(vals)
        audio_bytes = vals[np.argsort(ind)].astype(np.uint8).tobytes()

        if audio_bytes:
            data_origin, samplerate = sf.read(io.BytesIO(audio_bytes))

            # recordings may be mono or stereo depending on the device
            wav = data_origin[:, 0] if data_origin.ndim > 1 else data_origin

            data = librosa.resample(y=wav, orig_sr=samplerate, target_sr=TARGET_SR)

            if data.shape[0] > MIN_SECONDS * TARGET_SR:
                holder.empty()
                with st.spinner('Asking the Doc...'):
                    preprocessor = preprocess.AudioPreprocessor()
                    predictor = predict.MyPredictor(model, preprocessor)

                    # trim to a whole number of seconds
                    fs_mult = np.floor(data.shape[0] / TARGET_SR)
                    data = data[: int(TARGET_SR * fs_mult)]
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
                st.error("The recording must be at least 16 seconds long to obtain a result.")
