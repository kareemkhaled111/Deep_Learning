# Face Recognition Attendance System

## Project Overview

The **Face Recognition Attendance System** is a Flask-based web application that leverages computer vision and machine learning techniques to automate student attendance. The system captures and recognizes faces using a camera, trains a deep learning model to identify students, and records their attendance in a CSV log.

## Features

- **User Registration:** Allows new users (students) to register by providing details and capturing facial images.
- **Model Training:** The system trains a Convolutional Neural Network (CNN) to recognize faces from the captured images.
- **Face Recognition:** Identifies registered students in real-time using the webcam.
- **Attendance Logging:** Logs attendance when a student is recognized after multiple detections with high confidence.
- **Attendance View:** Displays a table of recorded attendance.

## Technology Stack

- **Backend:** Python, Flask
- **Computer Vision:** OpenCV, Haar Cascades
- **Deep Learning:** TensorFlow, Keras
- **Machine Learning:** Scikit-learn (for encoding labels and splitting data)
- **Database:** CSV (for storing user data and attendance records)
- **Other Libraries:** NumPy, Pandas, Pickle

## Requirements

Before running the project, ensure you have the following installed:

- Python 3.x
- Flask
- OpenCV
- TensorFlow
- NumPy
- Pandas
- Scikit-learn

You can install the required dependencies using the following command:

```bash

pip install -r requirements.txt

Project_Structure

/Attendance_Project
│
├── /static/                  # Static files (CSS, JS, images)
├── /templates/               # HTML templates
│   ├── index.html
│   ├── register.html
│   ├── success.html
│   ├── error.html
│   ├── model.html
│   └── view.html
├── /DATA/                    # Captured face images for each user
│   └── [username]/           # Folder for each user with face images
├── /logs/                    # Log files
│   └── attendance_log.csv    # Attendance log CSV file
├── app.py                    # Flask app entry point
├── face_recognition_model.h5 # Pre-trained model file
├── label_encoder.pkl         # Label encoder file
├── student_count.txt         # File to keep track of the number of students
└── user_data.csv             # CSV file with registered user data

1. Starting the Web Server
To run the Flask application, execute the following command:


python app.py
The application will start at http://127.0.0.1:5000/.

2. Registering a New User
Go to the Register page from the home page.
Provide the necessary user details (username, user ID, subject, email, etc.).
The system will capture facial images for registration.
After successful registration, the user will be added to the database and images will be stored in a separate folder.
3. Training the Model
After sufficient data is collected (images of different users), you can train the face recognition model.
Go to the Model Dashboard and click on Train Model.
The training process will begin and the model will be saved for future recognition tasks.
4. Testing the Model
After training, you can test the model by navigating to the Test page.
The system will attempt to recognize faces using your webcam and log attendance.
5. Viewing Attendance
Navigate to the View Attendance page to view the attendance log.
It displays the names of recognized students and the time they were marked present.
Model Details
The face recognition model is a Convolutional Neural Network (CNN) that uses the following architecture:

Convolutional Layers: Extract features from the face images.
MaxPooling Layers: Downsample the feature maps.
Flattening: Converts the 2D features into a 1D vector.
Fully Connected Layers: Classifies the face into one of the registered users.
Dropout: To prevent overfitting during training.
Files Overview
face_recognition_model.h5: The trained CNN model file for face recognition.
label_encoder.pkl: The label encoder that maps user labels to integers.
attendance_log.csv: CSV file where attendance is logged, including the name and time of recognition.
user_data.csv: CSV file containing user registration details (username, user ID, subject, etc.).
student_count.txt: A text file to keep track of the number of registered students.
Known Issues
The model might require tuning or retraining for better accuracy.
Ensure the dataset has sufficient images for each user to improve face recognition accuracy.
The system works best in a well-lit environment.
Future Improvements
Support for facial recognition under different lighting conditions.
Integration with external databases for user management.
Real-time alerts/notifications when a student is marked present.
Mobile app integration for remote attendance.



`
