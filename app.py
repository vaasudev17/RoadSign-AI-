import os
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image


# ============================================================
# 1. GTSRB CLASS MAPPING
# ============================================================

# IMPORTANT:
# This is the exact alphabetical folder order used during training.
class_names = [
    "0", "1", "10", "11", "12", "13", "14", "15",
    "16", "17", "18", "19", "2", "20", "21", "22",
    "23", "24", "25", "26", "27", "28", "29", "3",
    "30", "31", "32", "33", "34", "35", "36", "37",
    "38", "39", "4", "40", "41", "42", "5", "6", "7",
    "8", "9"
]


# ============================================================
# 2. READABLE TRAFFIC SIGN NAMES
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
# 3. LOAD MODEL
# ============================================================

print("Loading traffic sign model...")

model = tf.keras.models.load_model("traffic_sign_cnn.keras")

print("Model loaded successfully!")


# ============================================================
# 4. PREDICTION FUNCTION
# ============================================================

def predict_traffic_sign(image):
    if image is None:
        return "Please upload a traffic sign image.", ""

    try:
        # Convert image to RGB
        image = image.convert("RGB")

        # Resize exactly as during model prediction
        image = image.resize((128, 128))

        # Convert image to NumPy array
        img_array = np.array(image, dtype=np.float32)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # IMPORTANT:
        # DO NOT divide by 255 here.
        #
        # Your CNN already contains:
        # tf.keras.layers.Rescaling(1./255)

        prediction = model.predict(img_array, verbose=0)

        # Get the model output index
        output_index = int(np.argmax(prediction[0]))

        # Convert output index to actual GTSRB class number
        predicted_class = int(class_names[output_index])

        # Get readable class name
        predicted_name = class_names_readable[predicted_class]

        # Confidence
        confidence = float(prediction[0][output_index] * 100)

        return (
            f"🚦 {predicted_name}",
            f"Confidence: {confidence:.2f}%"
        )

    except Exception as e:
        return "Prediction failed.", str(e)


# ============================================================
# 5. GRADIO INTERFACE
# ============================================================

css = """
body {
    background: #f5f7fb;
}

.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}

.title {
    text-align: center;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 25px;
}

.result-box {
    text-align: center;
}
"""


with gr.Blocks(
    title="RoadSight AI - Traffic Sign Recognition",
    css=css
) as demo:

    gr.Markdown(
        """
        # 🚦 RoadSight AI
        ### Traffic Sign Recognition using Deep Learning

        Upload an image of a traffic sign and the trained CNN
        will identify the sign.
        """,
        elem_classes="title"
    )

    gr.Markdown(
        "The system displays the single best prediction from the 43 GTSRB traffic-sign classes.",
        elem_classes="subtitle"
    )

    with gr.Row():

        with gr.Column():
            image_input = gr.Image(
                type="pil",
                label="Upload Traffic Sign"
            )

            predict_button = gr.Button(
                "🔍 Predict Traffic Sign",
                variant="primary"
            )

        with gr.Column():

            prediction_output = gr.Textbox(
                label="Prediction",
                interactive=False
            )

            confidence_output = gr.Textbox(
                label="Confidence",
                interactive=False
            )

    predict_button.click(
        fn=predict_traffic_sign,
        inputs=image_input,
        outputs=[
            prediction_output,
            confidence_output
        ]
    )


# ============================================================
# 6. START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )