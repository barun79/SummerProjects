import pandas as pd

# Load data
df = pd.read_csv("./data/train.csv")

# Drop Loan_ID as it's not useful for prediction
df.drop(columns= ['Loan_ID'], inplace=True)

# Fill missing categorical values with mode
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']:
    mode_val = df[col].mode()[0]
    df[col].fillna(mode_val, inplace=True)

# Fill missing numerical value LoanAmount with median
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace= True)

print(df.isnull().sum())

# Save cleaned data to a new CSV
df.to_csv('./data/train_cleaned.csv', index=False)
print("Data cleaning complete, saved to train_cleaned.csv")