import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Model load karein
model = load_model('plant_disease_model.h5')

# Class names (apne dataset ke hisaab se yeh list honi chahiye)
class_names = ['Strawberry___Leaf_scorch', 'Squash___Powdery_mildew', 'Strawberry___healthy',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Corn_(maize)___Northern_Leaf_Blight', 'Pepper,_bell___healthy', 'Grape___healthy',
    'Blueberry___healthy', 'Grape___Esca_(Black_Measles)', 'Tomato___healthy', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Tomato___Septoria_leaf_spot', 'Tomato___Leaf_Mold',
    'Tomato___Bacterial_spot', 'Potato___healthy', 'Grape___Black_rot',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Soybean___healthy', 'Raspberry___healthy',
    'Apple___Cedar_apple_rust', 'Potato___Late_blight', 'Apple___healthy',
    'Corn_(maize)___Common_rust_', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Apple___Apple_scab', 'Tomato___Tomato_mosaic_virus', 'Tomato___Target_Spot',
    'Apple___Black_rot', 'Potato___Early_blight', 'Corn_(maize)___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Cherry_(including_sour)___Powdery_mildew', 'Peach___Bacterial_spot',
    'Cherry_(including_sour)___healthy']

def predict_disease(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.round(100 * np.max(prediction), 2)

    return f"Disease: {predicted_class}\nConfidence: {confidence}%"

interface = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🌿 Plant Disease Detector",
    description="Kisi bhi patte (leaf) ki photo upload karein, model bimari ya healthy status batayega."
)

interface.launch()
