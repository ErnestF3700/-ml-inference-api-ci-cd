# model/train.py
from sklearn.datasets import load_iris          
from sklearn.ensemble import RandomForestClassifier  
import joblib
import os                                       

def train() -> None:
    # 1. załaduj dane
    iris = load_iris()
    X, y = iris.data, iris.target               

    # 2. zbuduj i wytrenuj model
    clf = RandomForestClassifier(               
        n_estimators=100,                       
        random_state=42
    )
    clf.fit(X, y)

    # 3. zapisz model
    os.makedirs("model", exist_ok=True)         
    joblib.dump(clf, "model/iris_rf.joblib")    

if __name__ == "__main__":                      
    train()
