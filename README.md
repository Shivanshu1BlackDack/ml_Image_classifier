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

*(GIF of the webcam classifier in action)*

---

### Image Upload Mode
Upload any supported image format and get predictions instantly.

*(Screenshot of image upload with prediction)*

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
