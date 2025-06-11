from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("./data/train_cleaned.csv")

le = LabelEncoder()
df['Loan_Status'] = le.fit_transform(df['Loan_Status']) # Y -> 1, N - > 0

# One hot encoding remaining categorical features
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop('Loan_Status', axis = 1)
Y = df['Loan_Status']

# Spliting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state= 42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Initialize the model
model = LogisticRegression ()

# Train the model
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

# Evaluate performace
print("Accuracy: ", accuracy_score(Y_test, Y_pred))
print("Classification report", classification_report(Y_pred , Y_test))

# Save model and scaler
joblib.dump(model, './models/loan_approval_model.pkl')
joblib.dump(scaler, './models/scaler.pkl')
