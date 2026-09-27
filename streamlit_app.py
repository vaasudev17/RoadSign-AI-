import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="RoadSight AI",
    page_icon="🚦",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: "Space Grotesk", sans-serif;
}

.stApp {
    background: #071110;
    color: #e8f2ed;
}

.block-container {
    max-width: 1200px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}

.hero-title {
    font-size: 70px;
    font-weight: 700;
    line-height: 0.95;
    letter-spacing: -4px;
    margin-bottom: 25px;
}

.hero-subtitle {
    color: #91a9a0;
    font-size: 18px;
    line-height: 1.6;
    max-width: 600px;
}

.eyebrow {
    color: #b8ff62;
    font-family: "DM Mono", monospace;
    font-size: 12px;
    letter-spacing: 2px;
}

.model-chip {
    display: inline-block;
    margin-top: 25px;
    padding: 10px 18px;
    border: 1px solid #29423a;
    border-radius: 30px;
    color: #8ea79d;
    font-family: "DM Mono", monospace;
    font-size: 11px;
}

.result-card {
    background: #0b1916;
    border: 1px solid #29433a;
    border-radius: 18px;
    padding: 35px;
    margin-top: 30px;
}

.prediction {
    color: #b8ff62;
    font-size: 40px;
    font-weight: 700;
}

.confidence {
    font-size: 28px;
    font-weight: 700;
}

.small-label {
    color: #708980;
    font-family: "DM Mono", monospace;
    font-size: 10px;
    letter-spacing: 2px;
}

div.stButton > button {
    width: 100%;
    background: #b8ff62;
    color: #071110;
    border: none;
    border-radius: 10px;
    height: 55px;
    font-weight: 700;
    letter-spacing: 1px;
}

div.stButton > button:hover {
    background: #c8ff86;
    color: #071110;
}

[data-testid="stFileUploader"] {
    background: #0b1916;
    border: 1px dashed #38544a;
    border-radius: 14px;
    padding: 20px;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# CLASS NAMES
# --------------------------------------------------

class_names = {
    0: "Speed Limit 20 km/h",
    1: "Speed Limit 30 km/h",
    2: "Speed Limit 50 km/h",
    3: "Speed Limit 60 km/h",
    4: "Speed Limit 70 km/h",
    5: "Speed Limit 80 km/h",
    6: "End of Speed Limit 80 km/h",
    7: "Speed Limit 100 km/h",
    8: "Speed Limit 120 km/h",
    9: "No Passing",
    10: "No Passing for Vehicles Over 3.5 Tons",
    11: "Right-of-Way at Next Intersection",
    12: "Priority Road",
    13: "Yield",
    14: "Stop",
    15: "No Vehicles",
    16: "Vehicles Over 3.5 Tons Prohibited",
    17: "No Entry",
    18: "General Caution",
    19: "Dangerous Curve Left",
    20: "Dangerous Curve Right",
    21: "Double Curve",
    22: "Bumpy Road",
    23: "Slippery Road",
    24: "Road Narrows on Right",
    25: "Road Work",
    26: "Traffic Signals",
    27: "Pedestrians",
    28: "Children Crossing",
    29: "Bicycles Crossing",
    30: "Beware of Ice/Snow",
    31: "Wild Animals Crossing",
    32: "End of Speed and Passing Limits",
    33: "Turn Right Ahead",
    34: "Turn Left Ahead",
    35: "Ahead Only",
    36: "Go Straight or Right",
    37: "Go Straight or Left",
    38: "Keep Right",
    39: "Keep Left",
    40: "Roundabout Mandatory",
    41: "End of No Passing",
    42: "End of No Passing for Vehicles Over 3.5 Tons"
}


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model_path = "traffic_sign_cnn.keras"

    return tf.keras.models.load_model(model_path)


try:

    model = load_model()

except Exception as e:

    st.error("Could not load the traffic sign model.")

    st.code(str(e))

    st.stop()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="eyebrow">CNN TRAFFIC SIGN RECOGNITION</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-title">'
    'Turn a road sign<br>into a signal.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'Upload a traffic sign image and let our '
    'convolutional neural network identify it.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="model-chip">'
    '● &nbsp; CNN MODEL &nbsp;&nbsp; 128 × 128'
    '</div>',
    unsafe_allow_html=True
)


st.divider()


# --------------------------------------------------
# UPLOAD SECTION
# --------------------------------------------------

col1, col2 = st.columns([1, 1], gap="large")


with col1:

    st.markdown(
        '<div class="small-label">LIVE SCANNER</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Drop your traffic sign image here",
        type=["jpg", "jpeg", "png"]
    )


with col2:

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Traffic Sign",
            use_container_width=True
        )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    if st.button("SCAN SIGN"):

        with st.spinner("Analyzing traffic sign..."):

            try:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                # Resize exactly like training
                image = image.resize(
                    (128, 128)
                )

                # Convert to numpy
                image_array = np.array(
                    image,
                    dtype=np.float32
                )

                # Add batch dimension
                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                # Prediction
                prediction = model.predict(
                    image_array,
                    verbose=0
                )

                # Best prediction ONLY
                output_index = int(
                    np.argmax(prediction[0])
                )

                predicted_class = output_index

                predicted_name = class_names[
                    predicted_class
                ]

                confidence = float(
                    prediction[0][output_index] * 100
                )

                # --------------------------------------------------
                # RESULT
                # --------------------------------------------------

                st.markdown(
                    '<div class="result-card">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="small-label">'
                    'ANALYSIS COMPLETE'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="small-label" '
                    'style="margin-top:25px;">'
                    'PREDICTED TRAFFIC SIGN'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="prediction">'
                    f'{predicted_name}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="small-label" '
                    'style="margin-top:30px;">'
                    'MODEL CONFIDENCE'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="confidence">'
                    f'{confidence:.2f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.progress(
                    min(confidence / 100, 1.0)
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

            except Exception as e:

                st.error(
                    "Prediction failed."
                )

                st.code(str(e))


# --------------------------------------------------
# INFORMATION
# --------------------------------------------------

st.divider()

info1, info2, info3 = st.columns(3)


with info1:

    st.markdown("### 01 · See")

    st.write(
        "Upload a clear image of a traffic sign."
    )


with info2:

    st.markdown("### 02 · Infer")

    st.write(
        "The CNN analyzes the image and "
        "identifies the most likely traffic sign."
    )


with info3:

    st.markdown("### 03 · Explain")

    st.write(
        "View the predicted sign and "
        "model confidence."
    )


st.caption(
    "RoadSight AI · Deep Learning · Computer Vision · CNN"
)