import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

# ---------------------------
# Page Settings
# ---------------------------
st.set_page_config(
    page_title="Student Pass/Fail Predictor",
    page_icon="🎓",
    layout="centered"
)

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv("student_performance_dataset.csv")

df["Result"] = df["Result"].map({
    "Pass": 1,
    "Fail": 0
})

# Features and Target
X = df[[
    "Study_Time",
    "Attendance",
    "Assignment_Score",
    "Sleep_Hours"
]]

y = df["Result"]

# Train Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# ---------------------------
# UI
# ---------------------------

st.title("🎓 Student Pass/Fail Predictor")
st.write("Predict whether a student will Pass or Fail using KNN Machine Learning.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    study_time = st.number_input(
        "📚 Study Time (hours/day)",
        min_value=0.0,
        max_value=15.0,
        value=4.0
    )

    attendance = st.number_input(
        "🏫 Attendance (%)",
        min_value=0,
        max_value=100,
        value=75
    )

with col2:
    assignment_score = st.number_input(
        "📝 Assignment Score",
        min_value=0,
        max_value=100,
        value=60
    )

    sleep_hours = st.number_input(
        "😴 Sleep Hours",
        min_value=0.0,
        max_value=12.0,
        value=7.0
    )

st.markdown("---")

if st.button("🔍 Predict Result", use_container_width=True):

    input_data = pd.DataFrame(
        [[study_time, attendance, assignment_score, sleep_hours]],
        columns=[
            "Study_Time",
            "Attendance",
            "Assignment_Score",
            "Sleep_Hours"
        ]
    )

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Prediction: PASS")
        st.balloons()
    else:
        st.error("❌ Prediction: FAIL")

st.markdown("---")
st.caption("Built with Python, Scikit-Learn, KNN and Streamlit")