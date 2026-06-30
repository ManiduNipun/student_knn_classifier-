# student_knn_classifier-
This is an interactive Web Application built using Streamlit and Scikit-Learn that predicts whether a student will Pass or Fail an exam based on their daily habits and academic metrics. The core prediction engine uses the K-Nearest Neighbors (KNN) Classifier algorithm.


Key Features
Interactive UI: Users can dynamically input real-time metrics using smooth numeric input forms.

Feature Scaling: Seamlessly integrates StandardScaler to normalize raw data before computing distances, ensuring maximum KNN accuracy.

Instant ML Pipeline: Trains the KNN model instantly on background data and serves real-time predictions.

Modern Feedback Visuals: Uses celebratory UI components (like balloons and custom metric alerts) to notify users of a "Pass" or "Fail" verdict.

🛠️ Tech Stack & Libraries Used
Frontend Framework: Streamlit (For building the modern, interactive web dashboard)

Data Processing: Pandas & NumPy

Machine Learning: Scikit-Learn

KNeighborsClassifier (The core ML model)

StandardScaler (For normalizing distance parameters)

train_test_split (For validating dataset segments)




🏃‍♂️ How to Run This Project Locally

1. Clone the Repository

git clone https://github.com/your-username/student-knn-classifier.git
cd student-knn-classifier


2. Install Required Dependencies
Make sure you have Python installed, then run:

pip install streamlit pandas scikit-learn


3.Fire Up the Web App!

streamlit run app.py
