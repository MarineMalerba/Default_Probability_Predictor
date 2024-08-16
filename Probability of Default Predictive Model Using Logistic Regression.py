from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load the data from the uploaded CSV file into a DataFrame
file_path = #path to 'Loan_Data.csv'
df = pd.read_csv(file_path)

def predict_PD(credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, income, years_employed, fico_score):

    # Prepare data for Logistic Regression
    Y = df['default']
    X = df[['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, Y.values)

    # Create a DataFrame for the input with the same structure as X
    input_data = pd.DataFrame([[credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, income, years_employed, fico_score]], 
                              columns=X.columns)
    
    # Scale the input data before prediction
    input_scaled = scaler.transform(input_data)
    PD = model.predict_proba(input_scaled)

    return PD[0][1]

PD = predict_PD(4, 4224.542337, 7321.171391, 81534.51708, 5, 638)
print(f"Predicted Probability of Default: {PD:.4f}")
