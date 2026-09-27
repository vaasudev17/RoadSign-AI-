import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RoadSight AI",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');


/* ---------- MAIN ---------- */

.stApp {
    background: #071110;
    color: #e8f2ed;
}

.block-container {
    max-width: 1200px;
    padding-top: 40px;
    padding-bottom: 60px;
}


/* ---------- TEXT ---------- */

html,
body,
[class*="css"] {
    font-family: "Space Grotesk", sans-serif;
}


.eyebrow {
    color: #b8ff62;
    font-family: "DM Mono", monospace;
    font-size: 11px;
    letter-spacing: 2px;
    margin-bottom: 15px;
}


.hero-title {
    font-size: clamp(48px, 7vw, 82px);
    font-weight: 700;
    line-height: 0.95;
    letter-spacing: -4px;
    margin-bottom: 25px;
}


.hero-description {
    color: #91a9a0;
    font-size: 17px;
    line-height: 1.7;
    max-width: 600px;
}


.model-chip {
    display: inline-block;
    margin-top: 25px;
    padding: 11px 18px;
    border: 1px solid #29423a;
    border-radius: 30px;
    color: #8ea79d;
    font-family: "DM Mono", monospace;
    font-size: 11px;
}


.green {
    color: #b8ff62;
}


/* ---------- UPLOAD AREA ---------- */

.upload-title {
    color: #b8ff62;
    font-family: "DM Mono", monospace;
    font-size: 11px;
    letter-spacing: 2px;
    margin-bottom: 15px;
}


[data-testid="stFileUploader"] {
    background: #0b1916;
    border: 1px dashed #38544a;
    border-radius: 15px;
    padding: 20px;
}


[data-testid="stFileUploader"]:hover {
    border-color: #b8ff62;
}


/* ---------- BUTTON ---------- */

div.stButton > button {

    width: 100%;

    height: 55px;

    background: #b8ff62;

    color: #071110;

    border: none;

    border-radius: 10px;

    font-family: "Space Grotesk", sans-serif;

    font-weight: 700;

    letter-spacing: 1px;

    transition: 0.2s;
}


div.stButton > button:hover {

    background: #c8ff86;

    color: #071110;

    transform: translateY(-2px);

}


/* ---------- RESULT CARD ---------- */

.result-card {

    background: #0b1916;

    border: 1px solid #29433a;

    border-radius: 20px;

    padding: 35px;

    margin-top: 30px;

}


.result-label {

    color: #718980;

    font-family: "DM Mono", monospace;

    font-size: 10px;

    letter-spacing: 2px;

}


.prediction {

    color: #b8ff62;

    font-size: 42px;

    font-weight: 700;

    margin-top: 12px;

}


.confidence {

    color: #e8f2ed;

    font-size: 30px;

    font-weight: 700;

    margin-top: 10px;

}


/* ---------- INFO CARDS ---------- */

.info-card {

    background: #0b1916;

    border-top: 1px solid #29433a;

    padding: 25px;

}


.info-number {

    color: #b8ff62;

    font-family: "DM Mono", monospace;

    font-size: 11px;

}


.info-card h3 {

    font-size: 25px;

    margin-top: 15px;

}


.info-card p {

    color: #718980;

    line-height: 1.6;

}


/* ---------- FOOTER ---------- */

.footer {

    margin-top: 50px;

    padding-top: 25px;

    border-top: 1px solid #1b302b;

    color: #536b63;

    font-family: "DM Mono", monospace;

    font-size: 10px;

}


</style>
""", unsafe_allow_html=True)


# ============================================================
# GTSRB CLASS ORDER
# ============================================================
#
# IMPORTANT:
#
# Your training dataset folders were alphabetically sorted.
#
# Therefore:
#
# model output index != actual GTSRB class number
#
# Example:
#
# output index 6 -> folder "14" -> Stop
#
# This list reproduces the exact alphabetical ordering.
# ============================================================

class_names = [
    "0",
    "1",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "2",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "3",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
    "4",
    "40",
    "41",
    "42",
    "5",
    "6",
    "7",
    "8",
    "9"
]


# ============================================================
# HUMAN-READABLE TRAFFIC SIGN NAMES
# ============================================================

class_names_readable = {

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


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_traffic_model():

    model_path = "traffic_sign_cnn.keras"

    model = tf.keras.models.load_model(
        model_path
    )

    return model


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_traffic_model()

except Exception as error:

    st.error(
        "Unable to load the traffic sign model."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="eyebrow">'
    'CNN TRAFFIC SIGN RECOGNITION'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="hero-title">'
    'Turn a road sign<br>'
    'into a signal.'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="hero-description">'
    'Upload a traffic sign image and let our '
    'convolutional neural network identify it.'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="model-chip">'
    '<span class="green">●</span>'
    '&nbsp;&nbsp; CNN MODEL &nbsp;&nbsp;'
    '128 × 128'
    '</div>',
    unsafe_allow_html=True
)


st.write("")


st.divider()


# ============================================================
# UPLOAD SECTION
# ============================================================

left_column, right_column = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# LEFT SIDE
# ============================================================

with left_column:

    st.markdown(
        '<div class="upload-title">'
        'LIVE SCANNER'
        '</div>',
        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(
        "Upload a traffic sign image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        help="Upload a clear traffic sign image."
    )


# ============================================================
# RIGHT SIDE - IMAGE PREVIEW
# ============================================================

with right_column:

    if uploaded_file is not None:

        image_preview = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image_preview,
            caption="Uploaded Traffic Sign",
            use_container_width=True
        )


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    st.write("")


    if st.button(
        "SCAN SIGN",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Analyzing traffic sign..."
            ):

                # ------------------------------------------------
                # LOAD IMAGE
                # ------------------------------------------------

                image = Image.open(
                    uploaded_file
                ).convert("RGB")


                # ------------------------------------------------
                # EXACT SAME SIZE AS MODEL TRAINING
                # ------------------------------------------------

                image = image.resize(
                    (128, 128)
                )


                # ------------------------------------------------
                # CONVERT TO NUMPY
                # ------------------------------------------------

                img_array = np.array(
                    image
                )


                # ------------------------------------------------
                # ADD BATCH DIMENSION
                # ------------------------------------------------

                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )


                # ------------------------------------------------
                # PREDICTION
                # ------------------------------------------------
                #
                # IMPORTANT:
                #
                # DO NOT divide by 255 here.
                #
                # Your trained model already contains:
                #
                # Rescaling(1./255)
                #
                # ------------------------------------------------

                prediction = model.predict(
                    img_array,
                    verbose=0
                )


                # ------------------------------------------------
                # GET MODEL OUTPUT INDEX
                # ------------------------------------------------

                output_index = int(
                    np.argmax(
                        prediction[0]
                    )
                )


                # ------------------------------------------------
                # CONVERT OUTPUT INDEX TO ACTUAL
                # GTSRB CLASS NUMBER
                # ------------------------------------------------

                predicted_class = int(
                    class_names[
                        output_index
                    ]
                )


                # ------------------------------------------------
                # GET HUMAN-READABLE NAME
                # ------------------------------------------------

                predicted_class_name = (
                    class_names_readable[
                        predicted_class
                    ]
                )


                # ------------------------------------------------
                # CONFIDENCE
                # ------------------------------------------------

                confidence = float(
                    prediction[0][
                        output_index
                    ] * 100
                )


                # ==================================================
                # RESULT
                # ==================================================

                st.markdown(
                    '<div class="result-card">',
                    unsafe_allow_html=True
                )


                st.markdown(
                    '<div class="result-label">'
                    'ANALYSIS COMPLETE'
                    '</div>',
                    unsafe_allow_html=True
                )


                st.markdown(
                    '<div class="result-label" '
                    'style="margin-top:25px;">'
                    'PREDICTED TRAFFIC SIGN'
                    '</div>',
                    unsafe_allow_html=True
                )


                st.markdown(
                    f'<div class="prediction">'
                    f'{predicted_class_name}'
                    f'</div>',
                    unsafe_allow_html=True
                )


                st.markdown(
                    '<div class="result-label" '
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


                # ------------------------------------------------
                # CONFIDENCE BAR
                # ------------------------------------------------

                st.progress(
                    min(
                        confidence / 100,
                        1.0
                    )
                )


                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )


        except Exception as error:

            st.error(
                "Prediction failed."
            )

            st.code(
                str(error)
            )


# ============================================================
# INFORMATION SECTION
# ============================================================

st.write("")

st.divider()

st.write("")


info1, info2, info3 = st.columns(3)


with info1:

    st.markdown(
        '<div class="info-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="info-number">01</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<h3>See</h3>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a clear image of a traffic sign."
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


with info2:

    st.markdown(
        '<div class="info-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="info-number">02</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<h3>Infer</h3>',
        unsafe_allow_html=True
    )

    st.write(
        "The CNN analyzes the image and "
        "identifies the most likely traffic sign."
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


with info3:

    st.markdown(
        '<div class="info-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="info-number">03</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<h3>Explain</h3>',
        unsafe_allow_html=True
    )

    st.write(
        "View the predicted traffic sign "
        "and model confidence."
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'RoadSight AI · Deep Learning · '
    'Computer Vision · CNN'
    '</div>',
    unsafe_allow_html=True
)