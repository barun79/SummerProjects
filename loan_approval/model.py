from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("./data/train_cleaned.csv")

# Encode target variable
le = LabelEncoder()
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])  # Y -> 1, N -> 0

# One-hot encode categorical features
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Features and label
X = df.drop('Loan_Status', axis=1)
Y = df['Loan_Status']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # <- Use .transform here, not .fit_transform

# Model training
model = LogisticRegression()
model.fit(X_train, Y_train)

# Prediction
Y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))

# Save model and scaler
joblib.dump(model, 'loan_approval_model.sav')
joblib.dump(scaler, 'scaler.sav')
