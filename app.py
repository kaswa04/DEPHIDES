from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
import os
import re
import time
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import json
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



def load_data(file_path):
    # Load data into a list
    data = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split("\t", 1)  # Split at first space
            if len(parts) == 2:
                label, url = parts
                data.append((label, url))

    # Create DataFrame
    df = pd.DataFrame(data, columns=["Label", "URL"])
    return df



df_train = load_data("./train.txt")
X_train = df_train["URL"].values
tokenizer = Tokenizer(char_level=True, lower=False, oov_token=None)
tokenizer.fit_on_texts(X_train)

valid_urls = []

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key in production
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create uploads directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Create a directory for saved outputs
if not os.path.exists('outputs'):
    os.makedirs('outputs')

# Load the model (will be loaded when needed to save memory)
model = None

def load_model():
    global model
    if model is None:
        try:
            model = tf.keras.models.load_model('model\\my_cnn_model2.h5')
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    return True
    if model is None:
        try:
            model = tf.keras.models.load_model('model/url_classifier.h5')
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    return True

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def is_valid_url(url):
    # Basic URL validation regex
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None

def preprocess_url(url):
    # This is a placeholder for URL preprocessing
    # In a real application, you would extract features from the URL
    # For demonstration, we'll just return a simple vector
    # Assuming our model expects a feature vector of length 10
    if isinstance(url, str):  
        url = [url]  # Convert string to list

    # Convert URLs to numerical sequences
    # print(url)
    Pre_processed_url = tokenizer.texts_to_sequences(url)  
    # print(Pre_processed_url,"\n")
    # Set a fixed sequence length (padding/truncation)
    MAX_LEN = 200  # Adjust based on dataset analysis
    Pre_processed_url = pad_sequences(Pre_processed_url, maxlen=MAX_LEN, padding='post', truncating='post')

    return Pre_processed_url

def predict_url(url):
    # Load the model if not already loaded
    if not load_model():
        return {"error": "Failed to load model"}
    
    # Preprocess the URL
    # print(url,"\n")
    features = preprocess_url(url)
    # Make prediction
    # print("Features \n \n ",features)
    st_time = time.time()
    prediction = model.predict(features)
    end_time = time.time()
    # print("Prediction: ",prediction)
    # For demonstration, we'll simulate different prediction outcomes
    # In a real application, you would interpret the model's output
    result = {
        "url": url,
        "is_malicious": bool(prediction > 0.5),
        "prediction_time": f"{(end_time - st_time) * 1000:.4f} milliseconds"
    }
    
    return result


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input', methods=['GET', 'POST'])
def input_data():
    if request.method == 'POST':
        valid_urls.clear()
        # Check if the post request has the file part
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Read URLs from file
                with open(filepath, 'r') as f:
                    urls = [line.strip() for line in f if line.strip()]
                
                # Validate URLs
                
                invalid_urls = []
                for url in urls:
                    if is_valid_url(url):
                        valid_urls.append(url)
                    else:
                        invalid_urls.append(url)
                
                session['urls'] = valid_urls
                
                if invalid_urls:
                    flash(f'Found {len(invalid_urls)} invalid URLs in the file.', 'error')
                
                if valid_urls:
                    flash(f'Successfully loaded {len(valid_urls)} valid URLs from file.', 'success')
                    return redirect(url_for('index'))
                else:
                    flash('No valid URLs found in the file.', 'error')
            else:
                flash('Invalid file. Please upload a .txt file.', 'error')
        else:
            # Handle direct URL input
            url_input = request.form.get('url_input', '').strip()
            
            print("Received URL input:", url_input)  # ✅ Debugging

            if not url_input:
                flash('Please enter a URL.', 'error')
                return render_template('input.html')
            
            # Check if multiple URLs (one per line)
            urls = [u.strip() for u in url_input.split('\n') if u.strip()]
            
            invalid_urls = []
            for url in urls:
                if is_valid_url(url):
                    valid_urls.append(url)
                else:
                    invalid_urls.append(url)
            
            if invalid_urls:
                flash(f'Found {len(invalid_urls)} invalid URLs.', 'error')
            
            if valid_urls:
                session['urls'] = valid_urls
                flash(f'Successfully added {len(valid_urls)} valid URLs.', 'success')
                return redirect(url_for('index'))
            else:
                flash('No valid URLs provided.', 'error')
    
    return render_template('input.html')

@app.route('/process_url', methods=['POST'])
def process_url():
    data = request.get_json()
    url = data.get('url', '')
    # print("printinf current url",url)
    
    if not is_valid_url(url):
        return jsonify({"error": "Invalid URL format"})
    
    # Simulate processing time
    time.sleep(2)
    result = predict_url(url)
    return jsonify(result)

@app.route('/display')
def display_output():
    urls = session.get('urls', [])
    results = session.get('results', [])
    
    if not urls:
        flash('No URLs to process. Please enter URLs first.', 'error')
        return redirect(url_for('input_data'))
    
    return render_template('display.html', urls=urls, results=results)

@app.route('/store', methods=['GET', 'POST'])
def store_output():
    results = session.get('results', [])
    
    if not results:
        flash('No results to store. Please process URLs first.', 'error')
        return redirect(url_for('display_output'))
    
    if request.method == 'POST':
        filename = request.form.get('filename', 'output.txt')
        if not filename.endswith('.txt'):
            filename += '.txt'
        
        filepath = os.path.join('outputs', secure_filename(filename))
        
        with open(filepath, 'w') as f:
            for result in results:
                f.write(f"URL: {result['url']}\n")
                f.write(f"Malicious: {result['is_malicious']}\n")
                
                f.write(f"Prediction Time: {result['prediction_time']}\n")
                f.write("-" * 50 + "\n")

        session['saved_file'] = filepath
        
        flash(f'Results saved to {filename}', 'success')
        # return redirect(url_for('index'))
        return send_file(filepath, as_attachment=True)
    
    return render_template('store.html', results=results)


@app.route('/save_results', methods=['POST'])
def save_results():
    data = request.get_json()
    results = data.get('results', [])
    
    if results:
        session['results'] = results
        return jsonify({"status": "success"})
    else:
        return jsonify({"status": "error", "message": "No results to save"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)