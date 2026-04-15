import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    print("=" * 60)
    print("  AgriTech AI -- Crop Recommendation Model Training")
    print("=" * 60)

    # =========================================================
    # STEP 1: Data Collection
    # =========================================================
    print("\n[Step 1/7] Loading dataset...")
    csv_path = "../dataset/Crop_recommendation.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"  Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Features: {list(df.columns[:-1])}")
    print(f"  Target: '{df.columns[-1]}' -> {df['label'].nunique()} unique crops")

    os.makedirs("eda_results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # =========================================================
    # STEP 2: Exploratory Data Analysis (EDA)
    # =========================================================
    print("\n[Step 2/7] Running Exploratory Data Analysis...")

    # 2a. Dataset Overview
    print("\n--- Dataset Statistics ---")
    print(df.describe().round(2).to_string())
    print(f"\n  Missing values: {df.isnull().sum().sum()}")
    print(f"  Duplicate rows: {df.duplicated().sum()}")

    # 2b. Crop Distribution Plot
    plt.figure(figsize=(14, 6))
    crop_counts = df['label'].value_counts()
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(crop_counts)))
    bars = plt.bar(crop_counts.index, crop_counts.values, color=colors, edgecolor='white', linewidth=0.5)
    plt.title('Distribution of Crops in Dataset', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Crop Type', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    for bar, count in zip(bars, crop_counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                 str(count), ha='center', va='bottom', fontsize=8, fontweight='bold')
    plt.tight_layout()
    plt.savefig("eda_results/crop_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: eda_results/crop_distribution.png")

    # 2c. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, square=True, linewidths=1, linecolor='white',
                cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"})
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("eda_results/correlation_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: eda_results/correlation_heatmap.png")

    # 2d. Feature Distribution Plots
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for i, feature in enumerate(features):
        sns.histplot(df[feature], kde=True, ax=axes[i], color=colors[i * 3], edgecolor='white')
        axes[i].set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
        axes[i].set_xlabel(feature, fontsize=10)
        axes[i].set_ylabel('Count', fontsize=10)
    axes[-1].axis('off')  # Hide the 8th subplot
    fig.suptitle('Distribution of All Features', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("eda_results/feature_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: eda_results/feature_distributions.png")

    # 2e. Box Plots per Feature (showing crop variation)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()
    for i, feature in enumerate(features):
        # Select top 6 crops for readability
        top_crops = df['label'].value_counts().head(6).index
        subset = df[df['label'].isin(top_crops)]
        sns.boxplot(data=subset, x='label', y=feature, hue='label', ax=axes[i], palette='viridis', legend=False)
        axes[i].set_title(f'{feature} by Crop', fontsize=11, fontweight='bold')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_xlabel('')
    axes[-1].axis('off')
    fig.suptitle('Feature Variation Across Crops (Top 6)', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("eda_results/feature_boxplots.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: eda_results/feature_boxplots.png")

    # =========================================================
    # STEP 3: Data Preprocessing
    # =========================================================
    print("\n[Step 3/7] Preprocessing data...")

    # Features and Target
    X = df.drop("label", axis=1)
    y = df["label"]

    # Label Encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    joblib.dump(label_encoder, "models/label_encoder.joblib")
    print(f"  LabelEncoder fitted: {len(label_encoder.classes_)} classes")
    print(f"  Classes: {list(label_encoder.classes_)}")

    # Train Test Split (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"  Train/Test Split: {X_train.shape[0]} train / {X_test.shape[0]} test (80/20, stratified)")

    # Feature Scaling (StandardScaler / Z-score)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "models/scaler.joblib")
    print("  StandardScaler fitted and saved")

    # =========================================================
    # STEP 4: Model Training
    # =========================================================
    print("\n[Step 4/7] Training models...")

    # --- Model 1: Gaussian Naive Bayes ---
    print("\n  --- Model 1: Gaussian Naive Bayes ---")
    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)
    nb_preds = nb_model.predict(X_test_scaled)
    nb_acc = accuracy_score(y_test, nb_preds)
    print(f"  Naive Bayes Accuracy: {nb_acc * 100:.2f}%")

    # --- Model 2: Random Forest with GridSearchCV ---
    print("\n  --- Model 2: Random Forest (GridSearchCV) ---")
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf_base = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )

    print("  Running Hyperparameter Tuning (Grid Search with 5-Fold CV)...")
    grid_search.fit(X_train_scaled, y_train)

    best_rf = grid_search.best_estimator_
    print(f"\n  Best Parameters: {grid_search.best_params_}")

    rf_preds = best_rf.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"  Optimized Random Forest Accuracy: {rf_acc * 100:.2f}%")

    # =========================================================
    # STEP 5: Model Evaluation
    # =========================================================
    print("\n[Step 5/7] Evaluating models...")

    # --- K-Fold Cross Validation ---
    print("\n  --- K-Fold Cross Validation (5 Folds on Best RF) ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_rf, X_train_scaled, y_train, cv=kf, scoring='accuracy')
    print(f"  Fold Scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean CV Accuracy: {cv_scores.mean() * 100:.2f}% (+/-{cv_scores.std() * 100:.2f}%)")

    # --- Classification Reports ---
    print("\n  --- Classification Report: Naive Bayes ---")
    nb_report = classification_report(y_test, nb_preds, target_names=label_encoder.classes_)
    print(nb_report)

    print("\n  --- Classification Report: Random Forest ---")
    rf_report = classification_report(y_test, rf_preds, target_names=label_encoder.classes_)
    print(rf_report)

    # --- Save Classification Reports to file ---
    with open("eda_results/classification_reports.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  CLASSIFICATION REPORTS -- AgriTech AI\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Naive Bayes Accuracy: {nb_acc * 100:.2f}%\n")
        f.write("-" * 40 + "\n")
        f.write(nb_report + "\n\n")
        f.write(f"Random Forest Accuracy: {rf_acc * 100:.2f}%\n")
        f.write("-" * 40 + "\n")
        f.write(rf_report + "\n\n")
        f.write(f"Best RF Parameters: {grid_search.best_params_}\n")
        f.write(f"K-Fold CV Mean: {cv_scores.mean() * 100:.2f}% (+/-{cv_scores.std() * 100:.2f}%)\n")
    print("  Saved: eda_results/classification_reports.txt")

    # =========================================================
    # STEP 6: Visualization of Results
    # =========================================================
    print("\n[Step 6/7] Generating visualizations...")

    # 6a. Confusion Matrix -- Naive Bayes
    cm_nb = confusion_matrix(y_test, nb_preds)
    plt.figure(figsize=(16, 13))
    sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                linewidths=0.5, linecolor='white')
    plt.title(f'Confusion Matrix -- Naive Bayes (Accuracy: {nb_acc*100:.1f}%)',
              fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("eda_results/confusion_matrix_nb.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: eda_results/confusion_matrix_nb.png")

    # 6b. Confusion Matrix -- Random Forest
    cm_rf = confusion_matrix(y_test, rf_preds)
    plt.figure(figsize=(16, 13))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                linewidths=0.5, linecolor='white')
    plt.title(f'Confusion Matrix -- Random Forest (Accuracy: {rf_acc*100:.1f}%)',
              fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("eda_results/confusion_matrix_rf.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: eda_results/confusion_matrix_rf.png")

    # 6c. Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    models_names = ['Gaussian\nNaive Bayes', 'Random Forest\n(GridSearchCV)']
    accuracies = [nb_acc * 100, rf_acc * 100]
    bar_colors = ['#3498db', '#2ecc71']
    bars = ax.bar(models_names, accuracies, color=bar_colors, edgecolor='white',
                  linewidth=2, width=0.5)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.2,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.set_ylim(min(accuracies) - 2, 102)
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Model Comparison -- Accuracy', fontsize=16, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("eda_results/model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: eda_results/model_comparison.png")

    # 6d. Feature Importance (Random Forest)
    if isinstance(best_rf, RandomForestClassifier):
        importances = best_rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        feature_names = X.columns

        plt.figure(figsize=(10, 6))
        colors_fi = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
        y_labels = [feature_names[i] for i in indices]
        sns.barplot(
            x=importances[indices],
            y=y_labels,
            hue=y_labels,
            palette=[colors_fi[i] for i in range(len(indices))],
            legend=False
        )
        plt.title('Feature Importance in Crop Prediction (Random Forest)',
                  fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Relative Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        for i, (imp, idx) in enumerate(zip(importances[indices], indices)):
            plt.text(imp + 0.002, i, f'{imp:.3f}', va='center', fontsize=10)
        plt.tight_layout()
        plt.savefig("eda_results/feature_importance.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: eda_results/feature_importance.png")

    # 6e. Cross-Validation Score Plot
    plt.figure(figsize=(8, 5))
    fold_numbers = [f'Fold {i+1}' for i in range(len(cv_scores))]
    bars = plt.bar(fold_numbers, cv_scores * 100, color='#9b59b6', edgecolor='white', linewidth=2)
    plt.axhline(y=cv_scores.mean() * 100, color='#e74c3c', linestyle='--', linewidth=2,
                label=f'Mean: {cv_scores.mean()*100:.2f}%')
    for bar, score in zip(bars, cv_scores):
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                 f'{score*100:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.ylim(min(cv_scores * 100) - 2, 102)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('K-Fold Cross Validation Scores (5 Folds)', fontsize=14, fontweight='bold', pad=15)
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("eda_results/cross_validation_scores.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: eda_results/cross_validation_scores.png")

    # =========================================================
    # STEP 7: Save Best Model
    # =========================================================
    print("\n[Step 7/7] Saving best model...")

    best_model = best_rf if rf_acc >= nb_acc else nb_model
    best_name = "Random Forest" if rf_acc >= nb_acc else "Naive Bayes"
    joblib.dump(best_model, "models/best_model.joblib")
    print(f"  Best model saved: {best_name} -> models/best_model.joblib")

    # Save model metadata
    metadata = {
        "best_model": best_name,
        "best_accuracy": float(max(rf_acc, nb_acc)),
        "nb_accuracy": float(nb_acc),
        "rf_accuracy": float(rf_acc),
        "best_params": grid_search.best_params_,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "n_features": X.shape[1],
        "n_classes": len(label_encoder.classes_),
        "classes": list(label_encoder.classes_),
        "features": list(X.columns),
        "dataset_size": df.shape[0],
    }

    import json
    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("  Model metadata saved: models/model_metadata.json")

    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE -- Summary")
    print("=" * 60)
    print(f"  Dataset:               {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"  Crops:                 {len(label_encoder.classes_)} types")
    print(f"  Naive Bayes Accuracy:  {nb_acc * 100:.2f}%")
    print(f"  Random Forest Accuracy:{rf_acc * 100:.2f}%")
    print(f"  Best Model:            {best_name}")
    print(f"  K-Fold CV Mean:        {cv_scores.mean() * 100:.2f}% (+/-{cv_scores.std() * 100:.2f}%)")
    print(f"  Best RF Params:        {grid_search.best_params_}")
    print("=" * 60)
    print("\n  Files generated:")
    print("  - models/best_model.joblib")
    print("  - models/scaler.joblib")
    print("  - models/label_encoder.joblib")
    print("  - models/model_metadata.json")
    print("  - eda_results/crop_distribution.png")
    print("  - eda_results/correlation_heatmap.png")
    print("  - eda_results/feature_distributions.png")
    print("  - eda_results/feature_boxplots.png")
    print("  - eda_results/confusion_matrix_nb.png")
    print("  - eda_results/confusion_matrix_rf.png")
    print("  - eda_results/model_comparison.png")
    print("  - eda_results/feature_importance.png")
    print("  - eda_results/cross_validation_scores.png")
    print("  - eda_results/classification_reports.txt")
    print()


if __name__ == "__main__":
    main()
