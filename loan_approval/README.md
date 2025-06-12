
# 🏦 Loan Approval Prediction App

This is a Machine Learning-based web app built using **Streamlit** to predict loan approval outcomes based on user input data.

🔗 **Live App**: [Loan Approval App on Streamlit](https://summerprojects-4x5tzi2f6g5kmszqygghpk.streamlit.app/)

---

## 🚀 Features

- Predict loan approval based on applicant data
- Preprocessing with scaling and encoding
- Logistic Regression model
- Clean and interactive UI using Streamlit

---

## 📁 Project Structure

```
loan_approval/
├── app.py
├── loan_approval_model.sav
├── scaler.sav
├── requirements.txt
├── README.md
└── data/
    └── train_cleaned.csv
```

---

## ⚙️ How to Run Locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/barun79/SummerProjects.git
   cd SummerProjects/loan_approval
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## 💡 How the Model is Loaded

In `app.py`, the model and scaler are loaded with **safe relative paths**:

```python
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'loan_approval_model.sav'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.sav'))
```

> This ensures it works locally **and** when deployed (e.g., on Streamlit Cloud).

---

## 📦 Dependencies

Main libraries used:

- `pandas`
- `scikit-learn`
- `joblib`
- `streamlit`

See `requirements.txt` for full list.

---

## 🧠 Model Details

- **Algorithm**: Logistic Regression
- **Training data**: `data/train_cleaned.csv`
- **Preprocessing**:
  - Label encoding for target
  - One-hot encoding for categorical variables
  - StandardScaler for numeric features

---

## 👨‍💻 Author

Built by [Barun Singh](https://github.com/barun79)

---

## 📜 License

This project is licensed under the MIT License.
