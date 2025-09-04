# --- जरूरी लाइब्रेरीज इम्पोर्ट करें ---
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Path
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import cv2

# --- FastAPI ऐप को शुरू करें ---
app = FastAPI(
    title="Plant Disease API with JSON Mapping",
    description="यह API दो काम करती है: 1. इमेज से बीमारी का पता लगाना। 2. disease_key से बीमारी की जानकारी देना।"
)

# --- सर्वर शुरू होते ही मॉडल और डेटा लोड करें ---
try:
    MODEL =  tf.lite.Interpreter(model_path="cnn_model.tflite")
    
    CLASS_LABELS = [
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
        DISEASE_DATA = json.load(f)
        
    print("Model, class labels, and disease data loaded successfully!")

except Exception as e:
    print(f"Error loading resources: {e}")
    MODEL, CLASS_LABELS, DISEASE_DATA = None, [], None

# --- इमेज को प्रोसेस और प्रेडिक्ट करने के लिए हेल्पर फंक्शन ---
def process_and_predict(model, image_bytes):
    pil_image = Image.open(io.BytesIO(image_bytes))
    opencv_image = np.array(pil_image.convert('RGB'))
    img = cv2.resize(opencv_image, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Allocate tensors once
    model.allocate_tensors()

    # Get input/output details
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    # Set input
    model.set_tensor(input_details[0]['index'], img)

    # Run inference
    model.invoke()

    # Get output
    output_data = model.get_tensor(output_details[0]['index'])

    # Pick predicted class
    prediction = np.argmax(output_data, axis=-1)[0]
    return prediction


# --- एंडपॉइंट 1: इमेज अपलोड करके बीमारी का पता लगाएं ---
@app.post("/recognize-disease-from-image")
async def recognize_disease_from_image(file: UploadFile = File(...)):
    """
    यह एंडपॉइंट एक इमेज फाइल लेता है, मॉडल से प्रेडिक्शन करता है,
    और JSON से जानकारी निकालकर पूरा रिजल्ट देता है।
    """
    if not MODEL or not DISEASE_DATA:
        raise HTTPException(status_code=500, detail="Server resources not loaded.")

    image_bytes = await file.read()
    result_index = process_and_predict(MODEL, image_bytes)
    
    if result_index >= len(CLASS_LABELS):
        raise HTTPException(status_code=500, detail="Prediction index out of bounds.")
    
    predicted_key = CLASS_LABELS[result_index]
    
    for disease in DISEASE_DATA['plant_diseases']:
        if disease['disease_key'] == predicted_key:
            return disease
            
    raise HTTPException(status_code=404, detail=f"Info for key '{predicted_key}' not found.")

# --- एंडपॉइंट 2: JSON मैपिंग से बीमारी की जानकारी पाएं ---
@app.get("/get-disease-info/{disease_key}")
async def get_disease_info(disease_key: str = Path(..., description="JSON फाइल से बीमारी का की-वर्ड, जैसे 'Apple___Black_rot'")):
    """
    यह एंडपॉइंट एक `disease_key` लेता है और responses.json फाइल से
    उसकी पूरी जानकारी निकालकर देता है।
    """
    if not DISEASE_DATA:
        raise HTTPException(status_code=500, detail="Disease data not loaded.")

    for disease in DISEASE_DATA['plant_diseases']:
        if disease['disease_key'] == disease_key:
            return disease # अगर की मिल जाती है, तो उसकी जानकारी भेजें

    # अगर लूप खत्म हो गया और की नहीं मिली
    raise HTTPException(status_code=404, detail=f"'{disease_key}' नाम की कोई बीमारी नहीं मिली।")

# --- सर्वर को चलाने के लिए कोड ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)