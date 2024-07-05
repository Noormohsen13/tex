#!/usr/bin/env python
# coding: utf-8

# In[37]:


from flask import Flask, request, jsonify, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from scipy.ndimage import gaussian_filter, laplace
import tempfile
import cv2
import numpy as np

# Load the trained model
model = load_model('texture_transformed_model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

app = Flask(__name__)

# Function for texture transformations
def apply_texture_transformations(image):
    blurred_image = gaussian_filter(image, sigma=0.05)
    laplacian_image = laplace(blurred_image, mode='reflect') / 100.0
    noise = np.random.normal(0, 0.001, image.shape) * 255
    noisy_image = image + noise
    transformed_image = 0.98 * image + 0.01 * laplacian_image + 0.01 * noisy_image
    transformed_image = np.clip(transformed_image, 0, 255).astype(np.uint8)
    return transformed_image

# Function to resize image while maintaining quality using Bicubic Interpolation
def resize_image_with_quality(image, target_size):
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    return resized_image

# Function to load and preprocess external images while retaining original size
def load_and_preprocess_image(image_path):
    img = load_img(image_path)
    img_array = img_to_array(img)
    original_shape = img_array.shape[:2]
    resized_image = resize_image_with_quality(img_array, (224, 224))
    resized_image = resized_image.astype('float32') / 255.0
    return resized_image, original_shape

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image_path = temp.name
        file.save(image_path)

    image, original_shape = load_and_preprocess_image(image_path)
    if image is None:
        return jsonify({'error': 'Invalid image file'}), 400

    # Convert image to uint8 format (necessary for OpenCV)
    image_uint8 = (image * 255).astype(np.uint8)

    # Apply texture transformations
    transformed_image = apply_texture_transformations(image_uint8)

    # Resize transformed image to original size
    transformed_resized = resize_image_with_quality(transformed_image, original_shape[::-1])

    # Save transformed image with high quality
    transformed_image_path = image_path.replace(".jpg", "_transformed.jpg")
    cv2.imwrite(transformed_image_path, cv2.cvtColor(transformed_resized, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # Return transformed image as response with correct MIME type
    return send_file(transformed_image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


# In[39]:


from cryptography.fernet import Fernet

# Generate a Fernet key
fernet_key = Fernet.generate_key()
print(fernet_key.decode())


# In[41]:


# في ملف config.py
API_KEY = 'Sk4yTbilVZ6PBUUN7GozL63NhfmCaatwIAV-gu_efWo='


# In[48]:


from flask import Flask, request, jsonify, send_file
from functools import wraps
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from scipy.ndimage import gaussian_filter, laplace
import tempfile
import cv2
import numpy as np

app = Flask(__name__)

# Replace this with your actual API key
API_KEY = 'Sk4yTbilVZ6PBUUN7GozL63NhfmCaatwIAV-gu_efWo='

# Load the trained model
model = load_model('texture_transformed_model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Function for texture transformations
def apply_texture_transformations(image):
    blurred_image = gaussian_filter(image, sigma=0.05)
    laplacian_image = laplace(blurred_image, mode='reflect') / 100.0
    noise = np.random.normal(0, 0.001, image.shape) * 255
    noisy_image = image + noise
    transformed_image = 0.98 * image + 0.01 * laplacian_image + 0.01 * noisy_image
    transformed_image = np.clip(transformed_image, 0, 255).astype(np.uint8)
    return transformed_image

# Function to resize image while maintaining quality using Bicubic Interpolation
def resize_image_with_quality(image, target_size):
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    return resized_image

# Function to load and preprocess external images while retaining original size
def load_and_preprocess_image(image_path):
    img = load_img(image_path)
    img_array = img_to_array(img)
    original_shape = img_array.shape[:2]
    resized_image = resize_image_with_quality(img_array, (224, 224))
    resized_image = resized_image.astype('float32') / 255.0
    return resized_image, original_shape

# Function to check if the API key is valid
def check_api_key(api_key):
    return api_key == API_KEY

# Decorator function to enforce API key check
def require_api_key(view_func):
    @wraps(view_func)
    def decorated_function(*args, **kwargs):
        if 'API_KEY' not in request.headers or not check_api_key(request.headers['API_KEY']):
            return jsonify({'error': 'Unauthorized'}), 401
        return view_func(*args, **kwargs)
    return decorated_function

# Example route that requires API key
@app.route('/process_image', methods=['POST'])
@require_api_key
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image_path = temp.name
        file.save(image_path)

    image, original_shape = load_and_preprocess_image(image_path)
    if image is None:
        return jsonify({'error': 'Invalid image file'}), 400

    # Convert image to uint8 format (necessary for OpenCV)
    image_uint8 = (image * 255).astype(np.uint8)

    # Apply texture transformations
    transformed_image = apply_texture_transformations(image_uint8)

    # Resize transformed image to original size
    transformed_resized = resize_image_with_quality(transformed_image, original_shape[::-1])

    # Save transformed image with high quality
    transformed_image_path = image_path.replace(".jpg", "_transformed.jpg")
    cv2.imwrite(transformed_image_path, cv2.cvtColor(transformed_resized, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # Return transformed image as response with correct MIME type
    return send_file(transformed_image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


# In[6]:


get_ipython().system('python -m flask run --host=0.0.0.0 --port=5000')


# In[8]:


from flask_ngrok import run_with_ngrok

run_with_ngrok(app) 
app.run() 


# In[7]:


get_ipython().system('pip install flask-ngrok')
from flask_ngrok import run_with_ngrok

run_with_ngrok(app)  
app.run()


# In[ ]:




