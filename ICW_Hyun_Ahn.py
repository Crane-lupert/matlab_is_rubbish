# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Import necessary modules for EDA, preprocessing, modeling, and evaluation
from ydata_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import statsmodels.api as sm

# Define performance metric for GridSearchCV and evaluation
# User can choose one among: 'accuracy', 'f1', 'roc_auc', 'recall'
scoring_metric = 'f1'  # Change this variable as needed

# Define PCA explained variance threshold (e.g., 90% explanation)
pca_threshold = 0.70

# --------------
# 1. Data Loading and EDA on Raw Data
# --------------
# Load the raw dataset
raw_df = pd.read_csv("heart-disease.csv")

# Generate EDA report on raw data using ydata_profiling and save as HTML.
eda_report = ProfileReport(raw_df, title="Heart Disease EDA Report (Raw Data)", explorative=True)
eda_report.to_file("heart_disease_eda_raw.html")

# For further processing, work on a copy of the raw data.
df = raw_df.copy()

# --------------
# 2. Data Preprocessing
# --------------
# Binary encoding for 'famhist' column: 'Present' -> 1, 'Absent' -> 0.
df['famhist'] = df['famhist'].map({'Present': 1, 'Absent': 0})

# PCA Integration:
# Select features for PCA: 'adiposity', 'age', 'ldl', 'obesity', 'tobacco'
pca_features = ['adiposity', 'age', 'ldl', 'obesity', 'tobacco']
X_pca = df[pca_features]

# Standardize the PCA features
scaler_pca = StandardScaler()
X_pca_scaled = scaler_pca.fit_transform(X_pca)

# Apply PCA with explained variance threshold (n_components set as a float)
pca = PCA(n_components=pca_threshold, svd_solver='full')
X_pca_transformed = pca.fit_transform(X_pca_scaled)

# Print PCA results
print("PCA Explained Variance Ratio:", pca.explained_variance_ratio_)
print("PCA Cumulative Explained Variance:", pca.explained_variance_ratio_.cumsum())
print("PCA Components (Loadings):")
print(pca.components_)

# Drop the original PCA features and add the new PCA components to the dataframe
df = df.drop(columns=pca_features)
for i in range(X_pca_transformed.shape[1]):
    df[f'PC{i+1}'] = X_pca_transformed[:, i]

# Apply Min-Max Scaling to all feature columns except the target 'chd' and PCA columns (those starting with 'PC')
feature_columns = [col for col in df.columns if col != 'chd' and not col.startswith('PC')]
scaler_mm = MinMaxScaler()
df[feature_columns] = scaler_mm.fit_transform(df[feature_columns])

# --------------
# 3. Model Training - Logistic Regression with Ridge Penalty
# --------------
# Separate features and target variable
X = df.drop("chd", axis=1)
y = df["chd"]

# Split data into train and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train logistic regression with L2 (ridge) penalty
log_reg = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, C=1.0, random_state=42)
log_reg.fit(X_train, y_train)

# Predictions and evaluation for logistic regression
y_pred_log = log_reg.predict(X_test)
y_proba_log = log_reg.predict_proba(X_test)[:, 1]  # probability for class 1
log_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_log),
    'f1': f1_score(y_test, y_pred_log),
    'roc_auc': roc_auc_score(y_test, y_proba_log),
    'recall': recall_score(y_test, y_pred_log)
}
print("\nLogistic Regression (Ridge) Performance Metrics:")
print(log_metrics)
print("Coefficients:", log_reg.coef_)
print("Intercept:", log_reg.intercept_)

# For detailed coefficient statistics using statsmodels (without regularization)
X_train_sm = sm.add_constant(X_train)
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit(disp=0)
print(result.summary())

# --------------
# 4. Exploration of Other Classifiers
# --------------
# Define classifiers with specified hyperparameter grids for GridSearchCV (cv=5)
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=100, random_state=0),
    "Gradient Boosting": GradientBoostingClassifier(max_depth=3, n_estimators=100, random_state=0),
    "k-Nearest Neighbors": KNeighborsClassifier(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "GaussianNB": GaussianNB(),
    "SVC": SVC(probability=True, random_state=42)
}

param_grids = {
    "Decision Tree": {"ccp_alpha": [1, 0.1, 0.01, 0.001, 0.0001]},
    "Random Forest": {"max_features": [5, 10, 20, 30, 40, 50, "sqrt"]},
    "AdaBoost": {"learning_rate": [0.001, 0.01, 0.1, 1]},
    "Gradient Boosting": {"learning_rate": [0.001, 0.01, 0.1, 1]},
    "k-Nearest Neighbors": {"n_neighbors": [3, 5, 7, 9]},
    "LDA": {},
    "QDA": {},
    "GaussianNB": {},
    "SVC": {"kernel": ["rbf"], "gamma": [1, 1e-1, 1e-2, 1e-3, 1e-4], "C": [1, 10, 100, 1000]}
}

# Dictionary to store results for each classifier
results = {}

# Loop over classifiers and perform hyperparameter tuning with GridSearchCV (cv=5)
for name, clf in classifiers.items():
    print(f"\nTraining {name} classifier...")
    if param_grids[name]:
        grid = GridSearchCV(clf, param_grids[name], scoring=scoring_metric, cv=5)
        grid.fit(X_train, y_train)
        best_estimator = grid.best_estimator_
        best_params = grid.best_params_
        cv_score = grid.best_score_
        print(f"Best hyperparameters for {name}: {best_params}")
    else:
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring=scoring_metric)
        cv_score = scores.mean()
        best_estimator = clf
        best_estimator.fit(X_train, y_train)
        print(f"{name} cross-validated {scoring_metric}: {cv_score:.4f}")
    
    # Predict on test set
    y_pred = best_estimator.predict(X_test)
    # For classifiers with predict_proba, get probabilities for AUC; otherwise, use decision_function if available.
    if hasattr(best_estimator, "predict_proba"):
        y_proba = best_estimator.predict_proba(X_test)[:, 1]
    elif hasattr(best_estimator, "decision_function"):
        y_proba = best_estimator.decision_function(X_test)
    else:
        y_proba = None
    
    # Compute evaluation metrics for the classifier
    metrics_dict = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        'recall': recall_score(y_test, y_pred)
    }
    
    results[name] = {"CV Score": cv_score, "Test Metrics": metrics_dict, "Model": best_estimator}
    print(f"{name} Test Metrics:", metrics_dict)

# Define a separate metric for final model selection (e.g., accuracy)
selection_metric = 'accuracy'

# --------------
# 5. Selecting the Best Classifier
# --------------
# Define a separate metric for final model selection (e.g., accuracy)
# User can choose one among: 'accuracy', 'f1', 'roc_auc', 'recall'
selection_metric = 'accuracy'

# Select the best classifier based on the chosen selection_metric on the test set.
best_classifier_name = max(results, key=lambda k: results[k]["Test Metrics"][selection_metric] if results[k]["Test Metrics"][selection_metric] is not None else -np.inf)
best_classifier = results[best_classifier_name]["Model"]
best_metrics = results[best_classifier_name]["Test Metrics"]

print("\nBest classifier based on", scoring_metric, ":", best_classifier_name)
print("Best classifier Test Metrics:", best_metrics)

# classification report and confusion matrix for the best
y_best_pred = best_classifier.predict(X_test)
print("\nClassification Report for", best_classifier_name)
print(classification_report(y_test, y_best_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_best_pred))

