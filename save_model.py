"""
Same structure as notebook for Part 1 + new addition for streamlit
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

#question 2: defining cleaning function
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

#question 1: reading in 
s = pd.read_csv("social_media_usage.csv")

#question 3: creating target df
ss = pd.DataFrame({
    "sm_li":     clean_sm(s["web1h"]),
    "income":    np.where(s["income"] > 9, np.nan, s["income"]),
    "education": np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent":    clean_sm(s["par"]),
    "married":   clean_sm(s["marital"]),
    "female":    np.where(s["gender"] == 2, 1, 0),
    "age":       np.where(s["age"] > 98, np.nan, s["age"]),
}).dropna()

#question 4: defining target vector y and set X
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

#question 5: splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=987
)

#question 6: training model
lr = LogisticRegression(class_weight="balanced")
lr.fit(X_train, y_train)

#question 7: evaluating model and testing 
y_pred = lr.predict(X_test)
print("Rows after cleaning:", ss.shape)
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("Confusion matrix (TN, FP, FN, TP):", confusion_matrix(y_test, y_pred).ravel())

#now this is new addition
#"saves your trained model to a file so the Streamlit app can use"
joblib.dump(
    {
        #trained logistic regression (used for predictions)
        "model": lr,
        #column order the model expects
        "features": list(X.columns),
        #shown in the app footer
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        #sample size, shown in the app footer
        "n_rows": int(len(ss)),
    },
    "model.pkl",
)
print("Saved model.pkl")
