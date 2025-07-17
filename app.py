from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import traceback

app = Flask(__name__)

# Enable detailed error reporting
app.config['DEBUG'] = True

# Load trained model
try:
    model = load_model("alzheimers_model_trained.h5")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Upload path
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Class labels from your model
class_labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return jsonify({
        'status': 'Flask is working!',
        'model_loaded': model is not None,
        'upload_folder': UPLOAD_FOLDER,
        'upload_folder_exists': os.path.exists(UPLOAD_FOLDER)
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("📥 Received prediction request")
        
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded properly'}), 500
        
        # Check if file is in request
        if 'image' not in request.files:
            print("❌ No file part in request")
            return jsonify({'error': 'No file part in the request'}), 400

        img = request.files['image']
        print(f"📁 Received file: {img.filename}")
        
        if img.filename == '':
            print("❌ No selected file")
            return jsonify({'error': 'No selected file'}), 400

        if img:
            try:
                # Save uploaded image
                path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
                print(f"💾 Saving image to: {path}")
                img.save(path)
                
                # Verify file was saved
                if not os.path.exists(path):
                    return jsonify({'error': 'Failed to save uploaded file'}), 500
                
                print(f"✅ File saved successfully, size: {os.path.getsize(path)} bytes")

                # Load and preprocess image
                print("🔄 Loading and preprocessing image...")
                img_data = image.load_img(path, target_size=(128, 128))
                img_array = image.img_to_array(img_data)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                
                print(f"✅ Image preprocessed, shape: {img_array.shape}")

                # Predict
                print("🧠 Making prediction...")
                pred = model.predict(img_array)[0]
                predicted_class = class_labels[np.argmax(pred)]
                confidence = float(np.max(pred))
                
                print(f"✅ Prediction complete: {predicted_class} (confidence: {confidence:.2f})")

                # Clean up uploaded file (optional)
                try:
                    os.remove(path)
                    print("🗑️ Temporary file cleaned up")
                except:
                    print("⚠️ Could not clean up temporary file")

                return jsonify({
                    'prediction': predicted_class,
                    'confidence': f"{confidence:.2f}",
                    'all_predictions': {
                        class_labels[i]: f"{pred[i]:.3f}" 
                        for i in range(len(class_labels))
                    }
                })

            except Exception as e:
                print(f"❌ Error during image processing: {e}")
                print(f"📋 Traceback: {traceback.format_exc()}")
                return jsonify({
                    'error': f'Image processing failed: {str(e)}'
                }), 500

    except Exception as e:
        print(f"❌ Unexpected error in predict route: {e}")
        print(f"📋 Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

    return jsonify({'error': 'Prediction failed - unknown error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Flask application...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🧠 Model loaded: {model is not None}")
    app.run(debug=True, host='0.0.0.0', port=5000)