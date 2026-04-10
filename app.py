import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import dark_channel_prior as dcp
import inference as inf
from pycaret.classification import *
import seaborn as sns
import matplotlib.pyplot as plt

# Load YOLO model
MODEL_PATH = "Ocean_waste_Model_yolo11.pt"
model = YOLO(MODEL_PATH)

# Load Water Quality Model
WATER_MODEL_PATH = 'xgboost_without_source_month'
water_quality_model = load_model(WATER_MODEL_PATH)

# Water Quality Test Data
test_df = pd.read_csv('test_df')

# Waste Category Labels
labels = ['mask', 'can', 'cellphone', 'electronics', 'gbottle', 'glove', 'metal',
          'misc', 'net', 'pbag', 'pbottle', 'plastic', 'rod', 'sunglasses', 'tire']

# Helper Functions
def remove_noise(image):
    """Removes haze/noise from the input image using dark channel prior."""
    processed_image, _ = dcp.haze_removal(image, w_size=15, a_omega=0.95, gf_w_size=200, eps=1e-6)
    return processed_image

def detect_objects(image):
    """Performs object detection on the input image."""
    output_image, class_names = inf.detect(image)
    return output_image, class_names

def is_habitable(pH, Iron, Nitrate, Chloride, Lead, Zinc, Turbidity, Fluoride, Copper, Sulfate, Chlorine, Manganese,
                Total_Dissolved_Solids):
    if pH >= 6.5 and pH <= 9.0 and Iron < 0.3 and Nitrate < 10 and Chloride < 250 and Lead < 0.015 and Zinc < 5 and Turbidity < 5 and Fluoride >= 0.7 and Fluoride <= 1.5 and Copper < 1.3 and Sulfate < 250 and Chlorine < 4.0 and Manganese < 0.05 and Total_Dissolved_Solids < 500:
        return 0
    else:
        return 1

st.set_page_config(
    page_title="MarineCare",
    page_icon="🌊",
)

# Section Selector
selected_model = st.selectbox("Choose a page", 
                              ["Home", "Underwater Waste Detection Model", "Water Quality Assessment Model", "Generated Report"])

# Home Section
if selected_model == "Home":
    # Set the main title and description
    st.title("MarineCare: Marine Waste & Water Quality Detection 🌊🐟")

    st.subheader("Combating water pollution and ensuring water quality through advanced AI technologies.")

    st.write("""
    Let's safeguard marine ecosystems together! Neural Ocean provides:
    - **Underwater Waste Detection**: Using YOLOv11 to detect and classify waste materials in underwater images.
    - **Water Quality Assessment**: Evaluate water quality for aquatic life using machine learning models.
    """)

# Underwater Waste Detection Model
elif selected_model == "Underwater Waste Detection Model":
    st.title("Underwater Waste Detection Model")
    file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if file is not None:
        # Load and preprocess the image
        input_image = Image.open(file)
        input_image = np.array(input_image)
        st.image(input_image, caption="Uploaded Image")

        # Noise removal
        st.text("Removing noise from input...")
        input_image2 = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        st.image(input_image2, caption="Noise removed")

        # Object detection
        st.text("Detecting objects...")
        output_image, class_names = detect_objects(input_image2)
        st.image(output_image, caption="Detected Objects")

        if not class_names:
            st.success("The water is clear!")
        else:
            st.error(f"Waste Detected: {', '.join(class_names)}")

        # Save detected labels globally
        if "garbage" not in st.session_state:
            st.session_state.garbage = []
        st.session_state.garbage.extend(class_names)

# Water Quality Assessment Model
elif selected_model == "Water Quality Assessment Model":
    st.title('Water Quality Assessment Test')

    # Define the features and their types
    features = {
        "pH": float,
        "Iron": float,
        "Nitrate": float,
        "Chloride": float,
        "Lead": float,
        "Zinc": float,
        "Color": str,
        "Turbidity": float,
        "Fluoride": float,
        "Copper": float,
        "Odor": float,
        "Sulfate": float,
        "Chlorine": float,
        "Manganese": float,
        "Total Dissolved Solids": float,
    }

    inputs = {}
    col1, col2, col3 = st.columns(3)

    for i, feature in enumerate(features.items()):
        if i % 3 == 0:
            col = col1
        elif i % 3 == 1:
            col = col2
        else:
            col = col3
        with col:
            if feature[0] == "Color":
                options = ["Colorless", "Faint Yellow", "Light Yellow", "Near Colorless", "Yellow", "NaN"]
                inputs[feature[0]] = st.selectbox(f'{feature[0]}:', options=options, key=feature[0])
            else:
                inputs[feature[0]] = st.number_input(f'{feature[0]}:', value=0.0, step=0.1, format='%.1f', key=feature[0])

    # Add two buttons aligned in the center
    if st.button('Predict'):
        # Filter numerical features
        numerical_features = [
            "pH", "Iron", "Nitrate", "Chloride", "Lead", "Zinc", 
            "Turbidity", "Fluoride", "Copper", "Sulfate", 
            "Chlorine", "Manganese", "Total Dissolved Solids"
        ]
        inputs_list = [inputs[feature] for feature in numerical_features]  # Include only relevant features
        
        # Call is_habitable
        is_good = is_habitable(*inputs_list)
        if is_good == 0:
            st.success("Water quality is habitable for aquatic life")
        else:
            st.error("Water quality is not habitable for aquatic life")
        if "quality_aquatic" not in st.session_state:
            st.session_state.quality_aquatic = []
        st.session_state.quality_aquatic.append(is_good)

    if st.button('Random Inputs Predict'):
        data = test_df.sample(n=1)
        data.drop(['Target', 'Color', 'Odor'], axis=1, inplace=True)  # Drop non-numerical features
        st.write(data)
        is_good = is_habitable(*data.values.tolist()[0])
        if is_good == 0:
            st.success("Water quality is habitable for aquatic life")
        else:
            st.error("Water quality is not habitable for aquatic life")
        if "quality_aquatic" not in st.session_state:
            st.session_state.quality_aquatic = []
        st.session_state.quality_aquatic.append(is_good)


# Generated Report Section
elif selected_model == "Generated Report":
    st.title("Generated Report")

    # Frequency of Waste Labels
    st.header("Frequency of Detected Waste Categories")
    if "garbage" in st.session_state and st.session_state.garbage:
        occurrences = [st.session_state.garbage.count(label) for label in labels]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(y=labels, x=occurrences, palette="muted", ax=ax)
        ax.set_xlabel("Occurrences")
        ax.set_ylabel("Waste Categories")
        ax.set_title("Frequency of Detected Waste Categories")
        st.pyplot(fig)
    else:
        st.warning("No waste detection data available. Please run the Underwater Waste Detection Model.")

    # Water Quality for Aquatic Life Habitat
    st.header("Water Quality for Aquatic Life Habitat")
    if "quality_aquatic" not in st.session_state:
        st.session_state.quality_aquatic = []

    if st.session_state.quality_aquatic:
        counts = [st.session_state.quality_aquatic.count(0), st.session_state.quality_aquatic.count(1)]
        labels_h = ["Habitual", "Not Habitual"]
        colors = ['#cfaca4', '#623337']
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(counts, labels=labels_h, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title("Proportions of Water Quality for Aquatic Life Habitat")
        st.pyplot(fig)
    else:
        st.warning("No aquatic life habitat data available. Please run predictions in the Water Quality Assessment Model.")

    # Conclusion
    st.header("Conclusion")
    if "garbage" in st.session_state and st.session_state.garbage:
        occurrences = [st.session_state.garbage.count(label) for label in labels]
        most_common_waste = labels[occurrences.index(max(occurrences))] if any(occurrences) else "No waste detected"
    else:
        most_common_waste = "No waste data available"

    if "quality_aquatic" in st.session_state and st.session_state.quality_aquatic:
        aquatic_habitual = "Habitual" if st.session_state.quality_aquatic.count(0) > st.session_state.quality_aquatic.count(1) else "Not Habitual"
    else:
        aquatic_habitual = "No aquatic life habitat data available"

    # Display the conclusion summary
    st.markdown(f"""
    - **Most Common Waste Detected:** {most_common_waste}
    - **Aquatic Life Habitat Status:** {aquatic_habitual}
    """)
