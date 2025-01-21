import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Setup templates directory
templates = Jinja2Templates(directory="templates")

# Global variables
df = None
scaler = StandardScaler()
log_model = LogisticRegression()
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)

# Directory for uploads
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# 📌 Default route to render the homepage
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Render the homepage.
    """
    return templates.TemplateResponse("index.html", {"request": request})


# 📌 Endpoint to upload CSV file via form
@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    """
    Handle CSV file upload.
    """
    global df
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Save the file
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        message = f"File uploaded successfully. Columns: {list(df.columns)}"
    except Exception as e:
        message = f"Error uploading file: {str(e)}"

    return templates.TemplateResponse("index.html", {"request": request, "message": message})


# 📌 Endpoint to train models via form
@app.post("/train", response_class=HTMLResponse)
async def train(request: Request):
    """
    Train Logistic Regression and Decision Tree models.
    """
    global df, log_model, tree_model, scaler

    if df is None:
        return templates.TemplateResponse("index.html", {"request": request, "message": "No dataset uploaded!"})

    try:
        # Extract features and target variable
        X = df[['Temperature', 'Run_Time']]
        y = df['Downtime_Flag']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features for Logistic Regression
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train models
        log_model.fit(X_train_scaled, y_train)
        tree_model.fit(X_train, y_train)

        # Calculate metrics
        y_pred_log = log_model.predict(X_test_scaled)
        y_pred_tree = tree_model.predict(X_test)

        log_metrics = classification_report(y_test, y_pred_log, output_dict=True)
        tree_metrics = classification_report(y_test, y_pred_tree, output_dict=True)

        message = f"Training complete! Logistic Regression Accuracy: {log_metrics['accuracy']}, Decision Tree Accuracy: {tree_metrics['accuracy']}."
    except Exception as e:
        message = f"Error during training: {str(e)}"

    return templates.TemplateResponse("index.html", {"request": request, "message": message})


# 📌 Endpoint to predict downtime via form
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, Temperature: float = Form(...), Run_Time: float = Form(...)):
    """
    Predict downtime using input features.
    """
    global log_model, tree_model, scaler

    try:
        # Prepare input data
        input_data = np.array([[Temperature, Run_Time]])

        # Predict using Logistic Regression
        log_prediction = log_model.predict(scaler.transform(input_data))[0]
        log_confidence = log_model.predict_proba(scaler.transform(input_data))[0][log_prediction]

        # Predict using Decision Tree
        tree_prediction = tree_model.predict(input_data)[0]
        tree_confidence = tree_model.predict_proba(input_data)[0][tree_prediction]

        # Render results
        results = {
            "Logistic_Regression": {
                "Downtime": "Yes" if log_prediction == 1 else "No",
                "Confidence": round(log_confidence, 2)
            },
            "Decision_Tree": {
                "Downtime": "Yes" if tree_prediction == 1 else "No",
                "Confidence": round(tree_confidence, 2)
            }
        }

        return templates.TemplateResponse("result.html", {"request": request, "results": results})

    except Exception as e:
        message = f"Error during prediction: {str(e)}"
        return templates.TemplateResponse("index.html", {"request": request, "message": message})


# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
