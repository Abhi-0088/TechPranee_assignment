# Downtime Predictor Web Application

A simple web application built with FastAPI that allows users to:
- Upload a CSV dataset.
- Train Logistic Regression and Decision Tree models.
- Predict downtime based on input features (Temperature and Run Time).

The application includes a basic web interface for interaction.

---

## Features

1. **Upload Dataset**: Upload a CSV file containing the required columns (`Temperature`, `Run_Time`, `Downtime_Flag`).
2. **Train Models**: Train Logistic Regression and Decision Tree models on the uploaded dataset.
3. **Predict Downtime**: Use the trained models to predict downtime based on user-provided inputs.

---

## Project Structure

project/ 
│ 
├── main.py # FastAPI application 
├── templates/ 
│           ├── base.html # Base HTML template 
│           ├── index.html # Homepage for interaction 
│           ├── result.html # Results display page 
├── uploads/ # Directory to store uploaded files 
└── README.md # Project documentation


## Requirements

Ensure you have the following installed:
- Python 3.8 or higher
- Pip (Python package manager)

---

## Setup Instructions
git clone https://github.com/Abhi-0088/TechPranee_assignment.git
pip install fastapi uvicorn pandas numpy scikit-learn jinja2
python main.py
http://127.0.0.1:5000