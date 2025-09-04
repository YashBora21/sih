# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
from PIL import Image
import io # इमेज को मेमोरी में प्रोसेस करने के लिए

# --- 1. मॉडल को सिर्फ एक बार लोड करने के लिए कैशिंग का उपयोग करें ---
@st.cache_resource
def load_model_and_classes():
    """
    लोड करता है:
    1. प्रशिक्षित Keras मॉडल।
    2. क्लास के नामों की सूची (अंग्रेजी में)।
    3. JSON फाइल से हिंदी में बीमारी की जानकारी।
    """
    model = tf.keras.models.load_model('cnn_model.keras')
    
    class_labels = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]
    
    with open('responses.json', 'r', encoding='utf-8') as f:
        disease_data = json.load(f)
        
    return model, class_labels, disease_data

# --- 2. इमेज को प्रोसेस और प्रेडिक्ट करने के लिए फंक्शन ---
def predict_image(model, image_bytes):
    """
    अपलोड की गई इमेज को प्रोसेस करता है और बीमारी का इंडेक्स बताता है।
    """
    # Bytes को इमेज में बदलें
    pil_image = Image.open(io.BytesIO(image_bytes))
    opencv_image = np.array(pil_image.convert('RGB'))
    
    # इमेज को प्री-प्रोसेस करें
    H, W, C = 224, 224, 3
    img = cv2.resize(opencv_image, (H, W))
    img = img.astype('float32')
    img = img / 255.0
    img = img.reshape(1, H, W, C)

    # प्रेडिक्शन करें
    prediction = np.argmax(model.predict(img), axis=-1)[0]
    return prediction

# --- ऐप की शुरुआत में मॉडल और डेटा लोड करें ---
model, class_labels, disease_data = load_model_and_classes()


# --- Streamlit UI ---
st.sidebar.title('Plant Disease Prediction System')
app_mode = st.sidebar.selectbox('Select page', ['Home', 'Disease Recognition'])

if app_mode == 'Home':
    st.markdown("<h1 style='text-align: center;'>Plant Disease Prediction System for Sustainable Agriculture</h1>", unsafe_allow_html=True)
    try:
        home_img = Image.open('Disease.png')
        st.image(home_img, use_column_width=True)
    except FileNotFoundError:
        st.warning("Home page image ('Disease.png') not found.")

elif app_mode == 'Disease Recognition':
    st.header("Upload an Image for Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=['jpg', 'jpeg', 'png'])

    if test_image is not None:
        # --- 3. इमेज को फाइल में सेव करने की बजाय सीधे मेमोरी में प्रोसेस करें ---
        image_bytes = test_image.getvalue()
        
        # इमेज दिखाएं
        st.image(image_bytes, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Predict"):
            with st.spinner('Analyzing the image...'):
                # 1. मॉडल से प्रेडिक्शन इंडेक्स प्राप्त करें
                result_index = predict_image(model, image_bytes)
                
                # 2. इंडेक्स का उपयोग करके अंग्रेजी 'key' प्राप्त करें
                predicted_key = class_labels[result_index]
                
                # 3. 'key' का उपयोग करके सही बीमारी की जानकारी खोजें
                found_disease = None
                for disease in disease_data['plant_diseases']:
                    if disease['disease_key'] == predicted_key:
                        found_disease = disease
                        break
                
                # 4. हिंदी में परिणाम दिखाएं
                if found_disease:
                    st.success(f"**बीमारी का नाम:** {found_disease['disease_label']}")
                    
                    # 'healthy' केस को अलग से हैंडल करें
                    if 'healthy' in predicted_key:
                        st.write(found_disease['treatment_advice']['रोकथाम'])
                    else:
                        st.subheader("गंभीरता के लक्षण:")
                        st.write(f"**कम:** {found_disease['severity']['कम']}")
                        st.write(f"**मध्यम:** {found_disease['severity']['मध्यम']}")
                        st.write(f"**गंभीर:** {found_disease['severity']['गंभीर']}")
                        
                        st.subheader("उपचार और सलाह:")
                        st.write(f"**रोकथाम:** {found_disease['treatment_advice']['रोकथाम']}")
                        st.write(f"**शुरुआती इलाज:** {found_disease['treatment_advice']['शुरुआती इलाज']}")
                        st.write(f"**गंभीर इलाज:** {found_disease['treatment_advice']['गंभीर इलाज']}")
                else:
                    st.error("बीमारी की जानकारी नहीं मिली।")