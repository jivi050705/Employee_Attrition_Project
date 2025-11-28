# Employee_Attrition_Project

Project Description
This project builds an end‑to‑end machine learning pipeline to predict employee attrition using the IBM HR Analytics Employee Attrition dataset. The goal is to help HR teams identify employees who are at risk of leaving so they can take proactive retention actions. The solution covers data exploration, preprocessing, model training, evaluation, and business-friendly visualization.​

The workflow starts with exploratory data analysis on 1,470 employee records containing demographic, job, and satisfaction-related features, with attrition as the target variable. After cleaning the data and removing non-informative fields, categorical variables are encoded and the imbalanced target (≈16% attrition) is handled using SMOTE to balance the training set. Multiple classification algorithms are trained and compared, including Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost, with Gradient Boosting selected as the final model based on its overall performance.​

The best model achieves around 80% accuracy on the test set with a ROC-AUC of approximately 0.75, correctly identifying a significant share of employees who are likely to leave while maintaining strong performance on employees who stay. Evaluation includes a confusion matrix and attrition distribution plots to clearly show model behavior and the underlying class imbalance. Finally, model predictions are exported and used to build an interactive Power BI dashboard, where HR users can explore attrition risk by department, risk tier, and income level, and inspect a detailed table of high-risk employees for targeted interventions.​

How to Use This Project
1. Clone the repository
Clone the repo and navigate into the project folder:

git clone <your-repo-url>

cd <your-repo-folder>

2. Set up the environment
Create and activate a virtual environment (optional but recommended).

Install Python dependencies:

pip install -r requirements.txt

Make sure the IBM HR dataset CSV is placed in the Data/ folder with the expected file name.

3. Run the notebook
Open Employee_Attrition_Analysis.ipynb in Jupyter Notebook or VS Code.

Run the cells in order to:

Load and explore the dataset.

Preprocess and encode features.

Apply SMOTE to balance the training data.

Train multiple models and compare metrics.

Evaluate the final Gradient Boosting model with a confusion matrix and other metrics.

Generate prediction outputs and save them as employee_predictions.csv for visualization.​

4. Generate predictions for new data
Prepare a CSV file with the same structure as the training features (excluding the target).

Use the provided prediction function or script (e.g., predict_new_employees.py) to:

Load the saved model, scaler, and encoders.

Predict attrition labels and probabilities for new employees.

Save results (including risk category) to a new CSV file.

5. Build and use the Power BI dashboard
Open Power BI Desktop.

Import employee_predictions.csv:

Home → Get Data → Text/CSV → select file → Load.

Create visuals such as:

KPI cards (total employees, high/medium/low risk, predicted attritions).

Donut chart for risk distribution.

Bar chart for attrition by department.

Scatter chart of monthly income vs attrition probability.

High-risk employee table with key fields and probabilities.

Optionally, apply the provided dark-themed background image and formatting guidelines for a polished executive dashboard.

6. Reproduce or extend the work
Modify model hyperparameters or add new algorithms to improve performance.

Extend the dashboard with additional HR KPIs or drill-through pages.

Adapt the pipeline to other HR datasets by updating the preprocessing and feature configuration.
