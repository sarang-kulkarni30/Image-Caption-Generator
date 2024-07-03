from flask import Flask, render_template, request
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from pickle import load
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_features(image_path, model):
    try:
        image = Image.open(image_path)
    except:
        return None
    image = image.resize((299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        uploaded_file = request.files['file']
        
        if uploaded_file.filename != '':
            # Create the "uploads" directory if it doesn't exist
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            
            # Save the file to the "uploads" directory
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(image_path)
            
            # Extract features from the image
            xception_model = Xception(include_top=False, pooling="avg")
            photo = extract_features(image_path, xception_model)
            
            if photo is not None:
                # Load the pre-trained model and tokenizer
                tokenizer = load(open("tokenizer.p", "rb"))
                model = load_model('models/model_9.h5')
                
                # Generate the image caption
                description = generate_desc(model, tokenizer, photo, max_length=32)
                
                # Display the image and caption on the webpage
                image_base64 = image_to_base64(image_path)
                
                return render_template('index.html', caption=description, image_path=image_base64)
            else:
                return render_template('index.html', error="Error: Couldn't open image!")
    
    return render_template('index.html', caption=None, error=None, image_path=None)

if __name__ == '__main__':
    app.run(debug=True)