"""
predict_new_employees.py

Use the trained employee attrition model to predict attrition
for new employees from a CSV file.

Usage (from terminal):
    python predict_new_employees.py new_employees.csv
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib


def load_artifacts(model_path='employee_attrition_model.pkl',
                   scaler_path='scaler.pkl',
                   encoders_path='label_encoders.pkl'):
    """Load trained model, scaler, and label encoders."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    if not os.path.exists(encoders_path):
        raise FileNotFoundError(f"Encoders file not found: {encoders_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(encoders_path)

    return model, scaler, label_encoders


def preprocess_new_data(df, label_encoders):
    """
    Preprocess new employee data using the same steps as training:
    - drop unused columns if present
    - encode categorical columns with saved LabelEncoders
    - return numeric dataframe ready for scaling
    """

    data = df.copy()

    # Drop columns that were removed during training (if they exist)
    cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours', 'Attrition']
    for col in cols_to_drop:
        if col in data.columns:
            data = data.drop(columns=[col])

    # Encode categorical columns using the saved label_encoders
    for col, le in label_encoders.items():
        if col in data.columns:
            # Handle unseen categories by mapping them to -1
            def encode_value(x):
                if x in le.classes_:
                    return le.transform([x])[0]
                else:
                    return -1

            data[col] = data[col].map(encode_value)

    return data


def predict_new_employees(input_csv,
                          output_csv='new_employee_predictions.csv',
                          high_risk_csv='high_risk_employees.csv'):
    """Main function to predict attrition for new employees."""

    # Load new data
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    print(f"Loading new employee data from: {input_csv}")
    new_df = pd.read_csv(input_csv)
    original_df = new_df.copy()

    # Load model artifacts
    print("Loading model artifacts...")
    model, scaler, label_encoders = load_artifacts()

    # Preprocess
    print("Preprocessing new data...")
    X_new = preprocess_new_data(new_df, label_encoders)

    # Scale features
    print("Scaling features...")
    X_new_scaled = scaler.transform(X_new)

    # Predict
    print("Making predictions...")
    y_pred = model.predict(X_new_scaled)
    y_proba = model.predict_proba(X_new_scaled)[:, 1]

    # Build results dataframe
    results = original_df.copy()
    results['Attrition_Prediction'] = y_pred
    results['Attrition_Prediction_Label'] = results['Attrition_Prediction'].map({
        0: 'Will Stay',
        1: 'Will Leave'
    })
    results['Attrition_Probability'] = (y_proba * 100).round(2)

    # Risk categories
    results['Risk_Category'] = pd.cut(
        y_proba,
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )

    # Save full results
    results.to_csv(output_csv, index=False)
    print(f"Saved predictions for {len(results)} employees to: {output_csv}")

    # High-risk subset
    high_risk = results[results['Risk_Category'] == 'High Risk']
    if len(high_risk) > 0:
        high_risk.to_csv(high_risk_csv, index=False)
        print(f"High-risk employees ({len(high_risk)}) saved to: {high_risk_csv}")
    else:
        print("No high-risk employees found in this batch.")

    # Summary
    print("\nSummary:")
    print(results['Attrition_Prediction_Label'].value_counts())
    print("\nRisk distribution:")
    print(results['Risk_Category'].value_counts())

    return results


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict_new_employees.py <input_csv>")
        sys.exit(1)

    input_path = sys.argv[1]
    predict_new_employees(input_path)
