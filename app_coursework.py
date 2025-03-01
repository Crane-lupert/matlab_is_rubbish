# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shiny import ui, render, App
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif


#########################################
# Data Preparation and Processing Functions
#########################################

def prepare_datasets():
    """
    Loads the full credit card fraud dataset from 'creditcard.csv' and splits it into
    a test dataset and a training candidate dataset.

    - For each class (Class 0 and Class 1):
      * Randomly sample 50 observations to form the test set.
      * Remove these from the dataset; the remaining become the candidate for training.

    Returns:
      train_candidate: DataFrame with remaining samples after test extraction.
      test_df: DataFrame containing 50 samples from each class.
    """
    df = pd.read_csv("creditcard.csv")

    # Separate by class
    df_class0 = df[df["Class"] == 0]
    df_class1 = df[df["Class"] == 1]

    # Sample 50 observations from each class for test set
    test_class0 = df_class0.sample(n=50, random_state=42)
    test_class1 = df_class1.sample(n=50, random_state=42)
    test_df = pd.concat([test_class0, test_class1]).reset_index(drop=True)

    # Remove test samples for training candidate
    train_class0 = df_class0.drop(test_class0.index)
    train_class1 = df_class1.drop(test_class1.index)
    train_candidate = pd.concat([train_class0, train_class1]).reset_index(drop=True)

    return train_candidate, test_df


def create_strata(df, features):
    """
    For each feature in the given list, computes its median in df.
    For each sample, if the feature value is greater than or equal to the median, assign '1',
    otherwise assign '0'. The binary values for all selected features are concatenated
    (using underscores) to form a strata label.

    Example: For 5 features, a sample's strata might be "1_0_1_0_1".
    """
    strata_parts = []
    for feat in features:
        median_val = df[feat].median()
        binary = df[feat].apply(lambda x: "1" if x >= median_val else "0")
        strata_parts.append(binary)
    strata = pd.concat(strata_parts, axis=1).astype(str).agg('_'.join, axis=1)
    return strata


def undersample_class(df_class, target_count=200):
    """
    For a given class DataFrame (which contains a 'strata' column), perform undersampling
    based on the strata groups so that the total number of samples is target_count.
    Sampling is done proportionally within each strata group.
    """
    groups = []
    total = len(df_class)
    for name, group in df_class.groupby("strata"):
        n = int(round(len(group) / total * target_count))
        n = max(n, 1)
        n = min(n, len(group))
        groups.append(group.sample(n=n, random_state=42))
    result = pd.concat(groups)
    # Adjust if the total is not exactly target_count
    if len(result) > target_count:
        result = result.sample(n=target_count, random_state=42)
    elif len(result) < target_count:
        additional = df_class.drop(result.index).sample(n=target_count - len(result), random_state=42)
        result = pd.concat([result, additional])
    return result


def process_training_set(train_candidate):
    """
    Processes the training candidate dataset as follows:

    1. Splits the candidate into features (X) and target (y).
    2. Performs feature selection using SelectKBest (f_classif) to pick the top 10 important features.
    3. For the selected features, creates a strata label for each sample by applying a median threshold:
       if feature value >= median then '1', else '0'. The resulting binary values (for each of the 10 features)
       are concatenated (with underscores) to form a strata label.
    4. For each class (Class 0 and Class 1), performs undersampling based on the strata groups so that
       exactly 200 samples per class are obtained.

    Returns:
      train_final: Undersampled training dataset (400 samples total: 200 per class).
      selected_features: The list of top 10 selected features.
    """
    X_train = train_candidate.drop(columns=["Class"])
    y_train = train_candidate["Class"]

    # Feature Selection: select top 10 features using f_classif
    selector = SelectKBest(f_classif, k=10)
    selector.fit(X_train, y_train)
    selected_features = X_train.columns[selector.get_support()]
    print("Selected Features:", selected_features.tolist())

    # Create strata based on the selected features using median threshold
    strata = create_strata(X_train[selected_features], selected_features)
    train_candidate = train_candidate.copy()
    train_candidate["strata"] = strata

    # Undersample for each class to get exactly 200 samples per class
    train_class0 = train_candidate[train_candidate["Class"] == 0]
    train_class1 = train_candidate[train_candidate["Class"] == 1]

    train_class0_under = undersample_class(train_class0, target_count=200)
    train_class1_under = undersample_class(train_class1, target_count=200)

    train_final = pd.concat([train_class0_under, train_class1_under]).reset_index(drop=True)

    return train_final, selected_features


#########################################
# Shiny App Configuration and UI
#########################################

app_ui = ui.page_fluid(
    ui.h2("Credit Card Fraud Detection App"),
    ui.p("Select the model, adjust parameters, and choose whether to use Train-Test Split."),
    # Model selection: Decision Tree or Random Forest
    ui.input_select("model_type", "Select Model", {"dt": "Decision Tree", "rf": "Random Forest"}),
    # Toggle for Train-Test Split on/off
    ui.input_checkbox("use_split", "Use Train-Test Split", value=True),
    # Parameter slider for base parameter:
    # For DT: ccp_alpha; for RF: factor to compute max_features (0.0 ~ 0.2)
    ui.input_slider("param_value", "Parameter Value", 0.0, 0.2, 0.01, step=0.001),
    # Additional parameters:
    # For Decision Tree: max_depth
    ui.input_slider("max_depth", "Max Depth (DT)", 1, 20, 5, step=1),
    # For Random Forest: number of estimators
    ui.input_slider("n_estimators", "Number of Estimators (RF)", 50, 500, 100, step=10),
    # Output: Plot for visualization
    ui.output_plot("model_plot"),
    # Output: Text area for confusion matrices and accuracy info
    ui.output_text_verbatim("model_info")
)


#########################################
# Server Logic
#########################################

def server(input, output, session):
    @output
    @render.plot
    def model_plot():
        # Prepare datasets: get training candidate and test set
        train_candidate, test_df = prepare_datasets()

        # Decide: if use_split is ON, use separate training and test sets;
        # if OFF, combine them.
        if input.use_split():
            # Process training candidate to get final training set and selected features
            train_final, selected_features = process_training_set(train_candidate)
            # For visualization, use the first two features from the selected features.
            vis_features = list(selected_features)[:2]

            # Separate training set: drop "strata" and target
            X_train_full = train_final.drop(columns=["Class", "strata"])
            y_train_full = train_final["Class"]
            # For visualization, extract only the vis_features from training set
            X_train_vis = X_train_full[vis_features]

            # Test set remains as prepared (all features available)
            X_test_full = test_df.drop(columns=["Class"])
            y_test_full = test_df["Class"]
            # For visualization, take vis_features from test set
            X_test_vis = X_test_full[vis_features]
        else:
            # If split is OFF, combine training candidate and test set
            combined = pd.concat([train_candidate, test_df]).reset_index(drop=True)
            # Process combined dataset as training data
            train_final, selected_features = process_training_set(combined)
            vis_features = list(selected_features)[:2]
            X_train_full = train_final.drop(columns=["Class", "strata"])
            y_train_full = train_final["Class"]
            X_train_vis = X_train_full[vis_features]
            # In this case, testing is done on the same data
            X_test_full = X_train_full.copy()
            y_test_full = y_train_full.copy()
            X_test_vis = X_train_vis.copy()

        # Train the model on the full training data (using all selected features)
        if input.model_type() == "dt":
            # For Decision Tree, use ccp_alpha and max_depth from UI
            dt = DecisionTreeClassifier(ccp_alpha=input.param_value(),
                                        max_depth=input.max_depth(),
                                        random_state=42)
            dt.fit(X_train_full, y_train_full)
            # For visualization purposes, retrain a DT on the 2D projection (vis_features)
            dt_vis = DecisionTreeClassifier(ccp_alpha=input.param_value(),
                                            max_depth=input.max_depth(),
                                            random_state=42)
            dt_vis.fit(X_train_vis, y_train_full)

            # Create a meshgrid for decision boundary visualization
            x_min, x_max = X_train_vis.iloc[:, 0].min() - 1, X_train_vis.iloc[:, 0].max() + 1
            y_min, y_max = X_train_vis.iloc[:, 1].min() - 1, X_train_vis.iloc[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                 np.linspace(y_min, y_max, 200))
            Z = dt_vis.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            # Plot decision boundary
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
        else:
            # For Random Forest, use max_features and n_estimators from UI
            max_features_val = max(1, int(input.param_value() * X_train_full.shape[1]))
            rf = RandomForestClassifier(n_estimators=input.n_estimators(),
                                        max_features=max_features_val,
                                        random_state=42, oob_score=True)
            rf.fit(X_train_full, y_train_full)
            # For visualization, retrain on 2D projection
            max_features_vis = max(1, int(input.param_value() * X_train_vis.shape[1]))
            rf_vis = RandomForestClassifier(n_estimators=input.n_estimators(),
                                            max_features=max_features_vis,
                                            random_state=42, oob_score=True)
            rf_vis.fit(X_train_vis, y_train_full)

            # Create meshgrid for decision boundary visualization
            x_min, x_max = X_train_vis.iloc[:, 0].min() - 1, X_train_vis.iloc[:, 0].max() + 1
            y_min, y_max = X_train_vis.iloc[:, 1].min() - 1, X_train_vis.iloc[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                 np.linspace(y_min, y_max, 200))
            Z = rf_vis.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            # Plot decision boundary
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)

        # Plot scatter points:
        # If using split, plot train and test separately with different colors.
        if input.use_split():
            # Class 0: Train (sky blue), Test (navy)
            train_c0 = X_train_vis[y_train_full == 0]
            test_c0 = X_test_vis[y_test_full == 0]
            # Class 1: Train (orange), Test (yellow)
            train_c1 = X_train_vis[y_train_full == 1]
            test_c1 = X_test_vis[y_test_full == 1]

            ax.scatter(train_c0.iloc[:, 0], train_c0.iloc[:, 1], c='skyblue', label="Class 0 - Train", edgecolor='k')
            ax.scatter(test_c0.iloc[:, 0], test_c0.iloc[:, 1], c='navy', label="Class 0 - Test", marker='^',
                       edgecolor='k')
            ax.scatter(train_c1.iloc[:, 0], train_c1.iloc[:, 1], c='orange', label="Class 1 - Train", edgecolor='k')
            ax.scatter(test_c1.iloc[:, 0], test_c1.iloc[:, 1], c='yellow', label="Class 1 - Test", marker='^',
                       edgecolor='k')
        else:
            # If not splitting, plot all points with a single color per class.
            all_c0 = X_train_vis[y_train_full == 0]
            all_c1 = X_train_vis[y_train_full == 1]
            ax.scatter(all_c0.iloc[:, 0], all_c0.iloc[:, 1], c='blue', label="Class 0", edgecolor='k')
            ax.scatter(all_c1.iloc[:, 0], all_c1.iloc[:, 1], c='red', label="Class 1", edgecolor='k')

        ax.set_xlabel(vis_features[0])
        ax.set_ylabel(vis_features[1])
        ax.set_title("Decision Boundary & Data Scatter Plot")
        ax.legend()
        return fig

    @output
    @render.text
    def model_info():
        # Prepare datasets
        train_candidate, test_df = prepare_datasets()
        if input.use_split():
            # Use separate train and test sets
            train_final, selected_features = process_training_set(train_candidate)
            X_train_full = train_final.drop(columns=["Class", "strata"])
            y_train_full = train_final["Class"]
            X_test_full = test_df.drop(columns=["Class"])
            y_test_full = test_df["Class"]

            # Train models on training data and get predictions
            if input.model_type() == "dt":
                dt = DecisionTreeClassifier(ccp_alpha=input.param_value(),
                                            max_depth=input.max_depth(),
                                            random_state=42)
                dt.fit(X_train_full, y_train_full)
                y_train_pred = dt.predict(X_train_full)
                y_test_pred = dt.predict(X_test_full)
            else:
                max_features_val = max(1, int(input.param_value() * X_train_full.shape[1]))
                rf = RandomForestClassifier(n_estimators=input.n_estimators(),
                                            max_features=max_features_val,
                                            random_state=42, oob_score=True)
                rf.fit(X_train_full, y_train_full)
                y_train_pred = rf.predict(X_train_full)
                y_test_pred = rf.predict(X_test_full)

            # Compute confusion matrices
            cm_train = confusion_matrix(y_train_full, y_train_pred)
            cm_test = confusion_matrix(y_test_full, y_test_pred)
            # Compute accuracy for each set
            acc_train = accuracy_score(y_train_full, y_train_pred)
            acc_test = accuracy_score(y_test_full, y_test_pred)

            info = ("--- TRAINING SET ---\n"
                    f"Confusion Matrix:\n{cm_train}\n"
                    f"Accuracy: {acc_train:.2f}\n\n"
                    "--- TEST SET ---\n"
                    f"Confusion Matrix:\n{cm_test}\n"
                    f"Accuracy: {acc_test:.2f}")
        else:
            # Combine train_candidate and test set
            combined = pd.concat([train_candidate, test_df]).reset_index(drop=True)
            train_final, selected_features = process_training_set(combined)
            X_full = train_final.drop(columns=["Class", "strata"])
            y_full = train_final["Class"]

            if input.model_type() == "dt":
                dt = DecisionTreeClassifier(ccp_alpha=input.param_value(),
                                            max_depth=input.max_depth(),
                                            random_state=42)
                dt.fit(X_full, y_full)
                y_pred = dt.predict(X_full)
            else:
                max_features_val = max(1, int(input.param_value() * X_full.shape[1]))
                rf = RandomForestClassifier(n_estimators=input.n_estimators(),
                                            max_features=max_features_val,
                                            random_state=42, oob_score=True)
                rf.fit(X_full, y_full)
                y_pred = rf.predict(X_full)

            cm = confusion_matrix(y_full, y_pred)
            acc = accuracy_score(y_full, y_pred)
            info = ("--- FULL DATA (No Train-Test Split) ---\n"
                    f"Confusion Matrix:\n{cm}\n"
                    f"Accuracy: {acc:.2f}")
        return info


if __name__ == "__main__":
    # Create and run the Shiny app
    app = App(app_ui, server)
    app.run()