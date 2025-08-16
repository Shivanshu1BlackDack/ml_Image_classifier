# 🍽️ Real-time Food Recognition App

An **interactive web application** built with **Streamlit** that uses a **deep learning model** to classify images as **Food** or **Non-Food** in real time.  
You can use a **live webcam feed** or **upload an image** for classification.  
If an image is recognized as food, the model will further **categorize it into specific food types**.

---

## ✨ Features

- **📷 Live Webcam Classification** – Real-time predictions with your webcam.  
  - Green text → **Food**  
  - Red text → **Non-Food**
- **🖼️ Image Upload** – Upload images (`.jpg`, `.png`, `.jpeg`) for instant classification.
- **🍱 Multi-Class Food Categorization** – Detects categories like Bread, Dairy, Meat, Desserts, and more.
- **📊 Confidence Score** – Displays model confidence for each prediction.
- **💡 User-Friendly Interface** – Clean, intuitive UI built with Streamlit.

---

## 🚀 Demo

### Webcam Mode
Get instant results while pointing your camera at objects.  
> 🟩 Food is highlighted in green  
> 🟥 Non-Food is highlighted in red

<img width="600" height="400" alt="ml4" src="https://github.com/user-attachments/assets/acf9032a-e370-455b-be6d-4437fa57aa20" />

<img width="600" height="400" alt="ml6" src="https://github.com/user-attachments/assets/058ff166-9afc-437f-b2b5-08909e9de54d" />

---

### Image Upload Mode
Upload any supported image format and get predictions instantly.

<img width="600" height="400" alt="ml1" src="https://github.com/user-attachments/assets/d1666b24-3be8-4651-b225-3645fbdad217" />


---

## 🛠️ Technologies Used

- **Python 3.8+**
- **Streamlit** – Web app interface    
- **TensorFlow / Keras** – Deep learning model
- **OpenCV** – Webcam & image processing
- **Pillow (PIL)** – Image handling
- **NumPy** – Numerical computations

---

## ⚙️ Setup & Installation

### 1️⃣ Prerequisites
- Python **3.8 or newer**
- `pip` package manager

### 2️⃣ Install Dependencies
Create a `requirements.txt` file with:
streamlit
tensorflow
opencv-python-headless
numpy
Pillow

### ▶️ Run the Application

Run in terminal:

streamlit run app.py

(Replace app.py with your actual script name if different.)

Your default browser will open the application automatically.

### 🧠 Model Information

Type: Pre-trained CNN (Convolutional Neural Network)
Input Size: 224x224 pixels

Classes:

- Non-Food
- Bread
- Dairy product
- Dessert
- Egg
- Fried food
- Meat
- Noodles
- Rice
- Seafood
- Soup
- Vegetable

