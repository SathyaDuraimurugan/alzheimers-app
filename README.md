# 🧠 Alzheimer's Disease Prediction App

## Overview

**Alzheimer’s App** is a deep learning–based web application that predicts the stage of Alzheimer’s disease from brain MRI scans. Built using Convolutional Neural Networks (CNNs) and transfer learning techniques (VGG16, ResNet50, EfficientNet), the app classifies images into stages such as **Mild**, **Moderate**, **Severe**, or **Non-Demented**. Users can upload MRI images and receive instant predictions through an intuitive web interface.

---

## Features

- Upload MRI brain scans for diagnosis  
- Predict Alzheimer’s stage in real time  
- Uses CNN and pretrained models (VGG16, ResNet50, EfficientNet)  
- Simple and responsive frontend  
- Trained on Kaggle’s Alzheimer’s MRI dataset  
- Flask or Streamlit backend for model integration

---

## Tech Stack

### 🧠 Machine Learning
- Python, TensorFlow, Keras
- CNN architecture
- Transfer Learning (VGG16, ResNet50, EfficientNet)

### 🌐 Frontend
- HTML, CSS, JavaScript
- Bootstrap (optional)

### 🔧 Backend
- Flask / Streamlit
- Google Colab for training

---

## Project Structure

```
alzheimers-app/
├── model/                # Trained model files (.h5 or .pkl)
├── templates/            # HTML files
├── static/               # CSS and JS
├── app.py                # Flask/Streamlit backend
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/alzheimers-app.git
cd alzheimers-app
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
python app.py
```

Then open your browser and go to `http://localhost:5000`

---

## Dataset

The model is trained on the publicly available **Alzheimer’s MRI Dataset** from [Kaggle](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset).

---

## Future Improvements

- Include Grad-CAM visualizations  
- Add multi-image upload and batch predictions  
- Deploy using Heroku / Render / AWS  
- Add user login and history tracking  

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## Author

**Sathyaa Reddy**  
GitHub: [github.com/SathyaDuraimurugan](https://github.com/SathyaDuraimurugan)

