import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_psa_model():
    # 1. Load data
    df = pd.read_csv("data/processed/pulse_features.csv")
    X = df[['peak', 'rise_time', 'area', 'fwhm', 'chi_sq']]
    y = df['label']

    # 2. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. Train a Random Forest Classifier
    # We use RF because it handles non-linear separations well
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Feature Importance - which physical parameter mattered most?
    importances = model.feature_importances_
    for name, imp in zip(X.columns, importances):
        print(f"Feature: {name}, Importance: {imp:.4f}")

    # 5. Save the model for future use
    joblib.dump(model, "results/models/psa_random_forest.pkl")
    print("\nModel saved to results/models/psa_random_forest.pkl")

if __name__ == "__main__":
    train_psa_model()