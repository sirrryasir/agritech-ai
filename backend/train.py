import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("Loading dataset...")
    csv_path = "../dataset/Crop_recommendation.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    
    os.makedirs("eda_results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Features and Target
    X = df.drop("label", axis=1)
    y = df["label"]

    print("Preprocessing data...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    joblib.dump(label_encoder, "models/label_encoder.joblib")

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "models/scaler.joblib")

    print("\n--- Model 1: Gaussian Naive Bayes ---")
    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)
    nb_preds = nb_model.predict(X_test_scaled)
    nb_acc = accuracy_score(y_test, nb_preds)
    print(f"Naive Bayes Accuracy: {nb_acc * 100:.2f}%")

    print("\n--- Model 2: Advanced Random Forest (GridSearchCV) ---")
    # Define Parameter Grid for Hyperparameter Tuning
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    rf_base = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    
    print("Running Hyperparameter Tuning (Grid Search)...")
    grid_search.fit(X_train_scaled, y_train)
    
    best_rf = grid_search.best_estimator_
    print(f"Best Parameters Found: {grid_search.best_params_}")
    
    rf_preds = best_rf.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Optimized Random Forest Accuracy: {rf_acc * 100:.2f}%")

    print("\n--- Validating with K-Fold Cross Validation ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_rf, X_train_scaled, y_train, cv=kf, scoring='accuracy')
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean() * 100:.2f}%")

    # Save the best model
    best_model = best_rf if rf_acc >= nb_acc else nb_model
    print(f"\nSaving the best model to models/best_model.joblib")
    joblib.dump(best_model, "models/best_model.joblib")
    
    # Feature Importance Visualization
    if isinstance(best_model, RandomForestClassifier):
        print("\nExtracting Feature Importances...")
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = X.columns
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=[features[i] for i in indices], palette='viridis')
        plt.title('Feature Importance in Crop Prediction')
        plt.xlabel('Relative Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig("eda_results/feature_importance.png")
        plt.close()
        print("Feature importance graph saved to eda_results/feature_importance.png")

    print("\nTraining Complete! You now have an advanced, deeply-tuned AI model.")

if __name__ == "__main__":
    main()
