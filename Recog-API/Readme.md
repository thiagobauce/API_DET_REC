# Facial Recognition API
This is a Flask-based API that provides facial recognition services using the state-of-the-art models for face detection and recognition. The API allows users to upload images or videos and perform face detection and recognition tasks.

## Features
Face detection: The API utilizes a face detection program to detect faces in the uploaded images or videos.
Face recognition: The API includes a face recognition program to recognize faces in the uploaded images or videos.
Cropped images: The API saves the detected and recognized faces as cropped images in a directory with the current date.

## Getting Started
### Prerequisites
Python 3.x
Flask

### Installation

Clone the repository:
bash
!git clone https://github.com/thiagobauce/API_DET_REC.git


Install the required dependencies:
bash
!pip install flask


### Usage
Start the Flask development server:

python app.py

Open your web browser and access the API at http://localhost:5000.

Use the following endpoints:

GET /: Renders the index page with instructions on how to use the API.
POST /detect: Performs face detection on the uploaded image or video.
POST /recognize: Performs face recognition on the uploaded image or video.
GET /images_det: Displays the detected faces as cropped images.
GET /images_rec: Displays the recognized faces as cropped images.
Directory Structure
arduino
Copy code
├── app.py
├── Detection
│   └── detect.py
├── Recognition
│   └── arcface_torch
│       └── recognize.py
├── static
│   ├── css
│   │   └── style.css
│   └── js
│       └── script.js
└── templates
    ├── index.html
    ├── faces_detected.html
    └── faces_recognized.html

### Notes
The face detection and recognition programs (detect.py and recognize.py) are located in the Detection and Recognition/arcface_torch directories, respectively. Make sure to provide the correct paths to these programs in the API code.
The API saves the detected and recognized faces as cropped images in a directory with the current date. The images can be accessed through the /images_det and /images_rec endpoints.
Customize the HTML templates (index.html, faces_detected.html, and faces_recognized.html) as per your requirements.