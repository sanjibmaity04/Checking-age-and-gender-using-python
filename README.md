🧠 Age and Gender Detection using Python & Deep Learning
👤 Predict Age and Gender from Images or Live Camera using AI

🚀 Overview

This project uses deep learning and computer vision techniques to detect human faces and predict their age group and gender from static images or real-time webcam streams.

The system leverages OpenCV’s Deep Neural Network (DNN) module with pre-trained models for accurate and fast prediction.

🧩 Features

✅ Detects faces in real-time using OpenCV
✅ Predicts Gender: Male / Female
✅ Predicts Age Range (0–2, 4–6, 8–12, …, 60–100)
✅ Works on both images and webcam feed
✅ Lightweight & easy to deploy
✅ Modular design – ready for API or web integration

🧠 Tech Stack
Component	Technology Used
Programming Language	Python 3.x
Libraries	OpenCV, NumPy
Models	Pre-trained Caffe models (age_net.caffemodel, gender_net.caffemodel)
Framework	OpenCV DNN
Input Source	Image / Webcam
📁 Project Structure
📦 age-gender-detection
 ┣ 📂 model
 ┃ ┣ age_deploy.prototxt
 ┃ ┣ age_net.caffemodel
 ┃ ┣ gender_deploy.prototxt
 ┃ ┗ gender_net.caffemodel
 ┣ 📂 images
 ┃ ┗ sample.jpg
 ┣ main.py
 ┣ requirements.txt
 ┗ README.md
