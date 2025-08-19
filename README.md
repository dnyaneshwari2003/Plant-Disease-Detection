# 🌿 Plant Disease Detection using Deep Learning

This project leverages deep learning to classify plant leaf images and identify diseases with high accuracy. It uses TensorFlow/Keras to train a Convolutional Neural Network (CNN) on a labeled dataset of plant leaves, and provides a simple web interface using Flask for uploading and predicting plant disease in real-time.



## 📁 Project Structure

```
Plant-Disease-Detection/
├── static/
│   └── styles.css           # CSS styling for the web app
├── templates/
│   └── index.html           # Frontend HTML template
├── model/
│   └── Plant_Disease.h5     # Trained CNN model
├── app.py                   # Flask backend
├── dataset/                 # Dataset used for training
├── train_model.py           # CNN training script
└── requirements.txt         # Python dependencies
```


## 🚀 Features

* Image classification using Convolutional Neural Networks (CNNs)
* Detects and classifies plant diseases from leaf images
* Flask-based user-friendly web interface
* Trained on a custom dataset of healthy and diseased plant leaves
* Realtime prediction by uploading leaf images



## 🖼️ Supported Plant Diseases

Example labels (depending on dataset):

* Apple Scab
* Apple Black Rot
* Grape Leaf Blight
* Potato Late Blight
* Tomato Mosaic Virus
* Corn Common Rust
* Healthy (no disease)



## 🧠 Model Summary

* Model: CNN using Keras
* Layers: Conv2D, MaxPooling2D, Dropout, Flatten, Dense
* Trained using: Categorical Crossentropy Loss + Adam Optimizer
* Accuracy: \~98% on validation data



## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/manthan89-py/Plant-Disease-Detection.git
cd Plant-Disease-Detection
```

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask App

```bash
python app.py
```

Then open your browser and go to:
**[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

---

## 🌱 How to Use

1. Launch the app locally.
2. Upload a clear image of a plant leaf.
3. Click **Predict**.
4. The app will classify the disease and display the result instantly.

---

## 📊 Training

If you'd like to retrain the model:

```bash
python train_model.py
```

Make sure the `dataset/` folder is structured correctly with subfolders for each class.

---

## 📌 Requirements

* Python 3.6+
* TensorFlow / Keras
* Flask
* NumPy
* OpenCV (cv2)
* Pillow (PIL)

> All dependencies are listed in `requirements.txt`.

---

## 💡 Future Enhancements

* Add mobile support (PWA or Android app)
* Improve model robustness with more data
* Add confidence score and visual explanations (Grad-CAM)
* Cloud deployment (Heroku / AWS / Azure)

---

## Dataset: [PlantVillage Dataset](https://www.kaggle.com/emmarex/plantdisease)


---
