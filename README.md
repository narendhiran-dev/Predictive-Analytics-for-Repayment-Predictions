This file summarizes the project's purpose, technologies, and—most importantly—provides a clear, step-by-step guide with the exact commands needed to set up and run everything, incorporating all the lessons we learned from our debugging.

Here is the complete `README.md` file. You can copy and paste this entire block into the `README.md` file in your project folder.

---

# Borrower Repayment Prediction System

This project is a machine learning system that predicts a borrower's future repayment behavior based on their past payment history. It uses a `RandomForestRegressor` model served via a FastAPI REST API.

## Overview

The system analyzes a borrower's history of on-time, missed, and due payments to predict a "repayment percentage." Based on this percentage, it assigns a risk level (Low, Medium, or High). This provides valuable insights for investors or lenders about a borrower's reliability.

### Key Features
- **Predictive Model:** Utilizes a Scikit-learn `RandomForestRegressor` to predict future repayment performance.
- **Rich Feature Engineering:** Extracts features like payment ratios, delay statistics, and recent behavior trends.
- **Risk Classification:** Categorizes borrowers into Low, Medium, and High-risk tiers based on the model's prediction.
- **REST API:** Exposes the model's functionality through a clean, simple FastAPI endpoint.

## Technology Stack
- **Backend:** Python 3.11+
- **API Framework:** FastAPI
- **ML/Data Science:** Scikit-learn, Pandas, NumPy
- **Server:** Uvicorn

## Project Structure
```
repayment_prediction_system/
├── data/
│   ├── raw/
│   │   ├── payment_history.csv
│   │   └── investor_borrower.csv
│   └── processed/
├── repayment_predictor/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   ├── core/
│   │   └── config.py
│   ├── data_processing/
│   │   └── loader.py
│   ├── features/
│   │   └── build_features.py
│   └── models/
│       ├── predict_model.py
│       ├── risk_classifier.py
│       └── train_model.py
├── saved_models/
│   ├── random_forest_regressor.joblib
│   └── scaler.joblib
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Setup and Installation

Follow these steps to set up the project environment.

### 1. Prerequisites
- Python 3.11 or newer
- `pip` package manager

### 2. Clone the Repository
Download or clone this project to your local machine.
```bash
# Navigate to your desired directory
cd C:\path\to\your\projects

# Clone the repository (if it's in git)
# git clone ...
```

### 3. Create a Virtual Environment (Recommended)
From the project's root directory (`repayment_prediction_system`), create and activate a virtual environment.
```bash
# Create the virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\activate
```

### 4. Install Dependencies
Install all the required Python libraries.
```bash
pip install -r requirements.txt
```

### 5. Populate Data Files
The system requires sample data to run. Ensure the following files in `data/raw/` are not empty.

**`data/raw/payment_history.csv`**
```csv
payment_id,borrower_id,due_date,payment_date,status
1,1,2023-01-15,2023-01-14,on_time
2,1,2023-02-15,2023-02-18,missed
3,1,2023-03-15,2023-03-15,on_time
4,1,2023-04-15,2023-04-25,missed
5,1,2023-05-15,2023-05-28,missed
6,1,2023-06-15,2023-06-15,on_time
7,2,2023-01-10,2023-01-10,on_time
8,2,2023-02-10,2023-02-10,on_time
9,2,2023-03-10,2023-03-09,on_time
10,2,2023-04-10,2023-04-10,on_time
11,2,2023-05-10,2023-05-11,missed
12,3,2023-03-20,2023-05-20,missed
13,3,2023-04-20,2023-06-25,missed
14,3,2023-05-20,2023-08-01,missed
15,3,2023-06-20,,due
16,3,2023-07-20,,due
```

**`data/raw/investor_borrower.csv`**
```csv
investor_id,borrower_id
INV1,1
INV1,2
INV2,3
INV3,4
```

---

## Usage

Follow these steps in order to train the model and run the API.

### Step 1: Train the Model
Run the training script from the **project root directory**. This will process the data, train the model, and save the model artifacts (`.joblib` files) into the `saved_models/` directory.
```bash
python repayment_predictor\models\train_model.py
```
You should see output indicating the training process is complete and the model has been saved.

### Step 2: Run the API Server
Start the FastAPI server using Uvicorn.
```bash
uvicorn repayment_predictor.api.main:app --reload
```
The server will start and be available at `http://127.0.0.1:8000`. Keep this terminal window open.

### Step 3: Make a Prediction
Open a **new, second** terminal window to send a request to the running server.

To avoid command-line quoting issues on Windows, the most reliable method is to use a JSON file for the request body.

**A. Create a `request.json` file** in the project's root directory with the following content:
```json
{
  "investor_id": "INV2",
  "borrower_id": 3
}
```

**B. Send the request using `curl`**:
```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict-repayment" -H "Content-Type: application/json" -d @request.json
```

**Expected Successful Response:**
You will receive a JSON response with the prediction and summary for the requested borrower.
```json
{
  "borrower_repayment_summary": {
    "total_payments": 5,
    "on_time": 0,
    "missed": 3,
    "due": 2,
    "average_delay_days": 66.67,
    "max_delay_days": 73
  },
  "predicted_repayment_percentage": 11.96,
  "risk_level": "High Risk"
}
```

---

## API Endpoint Details

### POST `/predict-repayment`
This endpoint is used to predict the repayment behaviour of a borrower.

**Request Body:**
```json
{
  "investor_id": "string",
  "borrower_id": "integer"
}
```
- `investor_id` (string): The ID of the investor associated with the borrower.
- `borrower_id` (integer): The ID of the borrower whose repayment prediction is needed.

**Success Response (200 OK):**
```json
{
  "borrower_repayment_summary": {
    "total_payments": "integer",
    "on_time": "integer",
    "missed": "integer",
    "due": "integer",
    "average_delay_days": "float",
    "max_delay_days": "integer"
  },
  "predicted_repayment_percentage": "float",
  "risk_level": "string"
}
```
