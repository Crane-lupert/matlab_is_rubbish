#!pip install kagglehub
#!pip install shiny
#!pip install mplcursors

import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from shiny import ui, render, App
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
import mplcursors

# Data Loading & Preprocessing
path = kagglehub.dataset_download("dhanushnarayananr/credit-card-fraud")
print("Path to dataset files:", path)
df = pd.read_csv(path + "/card_transdata.csv")
df_class0 = df[df["fraud"] == 0].sample(n=200, random_state=42)
df_class1 = df[df["fraud"] == 1].sample(n=200, random_state=42)
df_balanced = pd.concat([df_class0, df_class1]).reset_index(drop=True)
from scipy.stats import skew
print("Skewness of distance_from_home:", skew(df_balanced["distance_from_home"]))
print("Skewness of distance_from_last_transaction:", skew(df_balanced["distance_from_last_transaction"]))
df_balanced["distance_from_home"] = np.log1p(df_balanced["distance_from_home"])
df_balanced["distance_from_last_transaction"] = np.log1p(df_balanced["distance_from_last_transaction"])
scaler_std = StandardScaler()
df_balanced["ratio_to_median_purchase_price"] = scaler_std.fit_transform(df_balanced[["ratio_to_median_purchase_price"]])
X = df_balanced.drop(columns=["fraud"])
y = df_balanced["fraud"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train Set Fraud 비율:\n", y_train.value_counts(normalize=True))
print("Test Set Fraud 비율:\n", y_test.value_counts(normalize=True))

# Helper: Disabled slider HTML
def disabled_slider(label, min_val, max_val, value, step):
    html_str = f'''
    <label>{label}</label><br>
    <input type="range" min="{min_val}" max="{max_val}" value="{value}" step="{step}" disabled style="width:100%"><br>
    '''
    return ui.HTML(html_str)

# Define UI with dynamic conditional UI components.
app_ui = ui.page_fluid(
    ui.h2("Credit Card Fraud Detection App"),
    ui.p("Select the model, adjust parameters, and choose whether to use Train-Test Split."),
    ui.input_select("model_type", "Select Model", {"dt": "Decision Tree", "rf": "Random Forest"}),
    ui.input_checkbox("use_split", "Use Train-Test Split", value=True),
    # 두 개의 파라미터 슬라이더 분리: DT용과 RF용
    ui.output_ui("parameter_slider_ui"),
    ui.p("For DT: 'ccp_alpha (DT)' controls tree complexity. For RF: 'Base Factor (RF)' is used to compute max_features."),
    # 조건부 슬라이더: DT 전용은 Max Depth, RF 전용은 Number of Estimators
    ui.output_ui("conditional_sliders"),
    # 동적 x축 선택 메뉴
    ui.output_ui("xaxis_choice_ui"),
    ui.output_plot("model_plot"),
    ui.output_text_verbatim("model_info"),
    ui.output_ui("model_eval")
)

def server(input, output, session):

    @output
    @render.ui
    def parameter_slider_ui():
        if input.model_type() == "dt":
            return ui.TagList(
                ui.input_slider("dt_ccp_alpha", "ccp_alpha (DT)", 0.0, 0.2, 0.01, step=0.001),
                disabled_slider("Base Factor (RF)", 0.0, 0.2, 0.01, 0.001)
            )
        elif input.model_type() == "rf":
            return ui.TagList(
                disabled_slider("ccp_alpha (DT)", 0.0, 0.2, 0.01, 0.001),
                ui.input_slider("rf_base_factor", "Base Factor (RF)", 0.0, 0.2, 0.01, step=0.001)
            )
        else:
            return ui.TagList()

    @output
    @render.ui
    def conditional_sliders():
        if input.model_type() == "dt":
            return ui.TagList(
                ui.input_slider("max_depth", "Max Depth (DT)", 1, 20, 5, step=1),
                ui.p("Maximum depth of the tree (limits how deep the tree grows)."),
                disabled_slider("Number of Estimators (RF)", 50, 500, 100, 10)
            )
        elif input.model_type() == "rf":
            return ui.TagList(
                disabled_slider("Max Depth (DT)", 1, 20, 5, 1),
                ui.input_slider("n_estimators", "Number of Estimators (RF)", 50, 500, 100, step=10),
                ui.p("Number of trees in the forest (RF).")
            )
        else:
            return ui.TagList()

    @output
    @render.ui
    def xaxis_choice_ui():
        if input.model_type() == "dt":
            return ui.input_select("xaxis_choice", "Select X-axis for Accuracy Curve",
                                     {"dt_ccp_alpha": "ccp_alpha (DT)", "max_depth": "Max Depth (DT)"})
        elif input.model_type() == "rf":
            return ui.input_select("xaxis_choice", "Select X-axis for Accuracy Curve",
                                     {"rf_base_factor": "Base Factor (RF)", "n_estimators": "Number of Estimators (RF)"})
        else:
            return ui.TagList()

    def build_confusion_metrics_table(y_true, y_pred, y_prob):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1_val = f1_score(y_true, y_pred)
        aucroc = roc_auc_score(y_true, y_prob)
        acc_html = f'<span title="Accuracy: The proportion of total predictions that were correct.">{acc:.4f}</span>'
        prec_html = f'<span title="Precision: The proportion of positive predictions that were correct.">{prec:.4f}</span>'
        rec_html = f'<span title="Recall: The proportion of actual positives correctly predicted.">{rec:.4f}</span>'
        f1_html = f'<span title="F1-Score: The harmonic mean of precision and recall.">{f1_val:.4f}</span>'
        auc_html = f'<span title="AUC-ROC: The Area Under the ROC Curve.">{aucroc:.4f}</span>'
        data_cm = {"Predicted 0": [tn, fn],
                   "Predicted 1": [fp, tp]}
        index_cm = ["Actual 0", "Actual 1"]
        df_cm = pd.DataFrame(data_cm, index=index_cm)
        data_eval = {"Metric": ["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"],
                     "Value": [acc_html, prec_html, rec_html, f1_html, auc_html]}
        df_eval = pd.DataFrame(data_eval)
        html_out = df_cm.to_html(escape=False, index=True)
        html_out += "<br>" + df_eval.to_html(escape=False, index=False)
        return html_out

    def train_model(model_type, split, dt_ccp_alpha, rf_base_factor, max_depth, n_estimators):
        if split:
            X_train_full = X_train.reset_index(drop=True)
            y_train_full = y_train.reset_index(drop=True)
            X_test_full = X_test.reset_index(drop=True)
            y_test_full = y_test.reset_index(drop=True)
            X_train_vis = TSNE(n_components=2, random_state=42).fit_transform(X_train_full)
            X_train_vis = pd.DataFrame(X_train_vis, columns=['TSNE1', 'TSNE2']).reset_index(drop=True)
            X_test_vis = TSNE(n_components=2, random_state=42).fit_transform(X_test_full)
            X_test_vis = pd.DataFrame(X_test_vis, columns=['TSNE1', 'TSNE2']).reset_index(drop=True)
        else:
            X_train_full = X.reset_index(drop=True)
            y_train_full = y.reset_index(drop=True)
            X_test_full = X.reset_index(drop=True)
            y_test_full = y.reset_index(drop=True)
            X_train_vis = TSNE(n_components=2, random_state=42).fit_transform(X_train_full)
            X_train_vis = pd.DataFrame(X_train_vis, columns=['TSNE1', 'TSNE2']).reset_index(drop=True)
            X_test_vis = X_train_vis.copy()
        if model_type == "dt":
            param_val = dt_ccp_alpha
            max_depth_val = max_depth if max_depth is not None else 5
            model = DecisionTreeClassifier(ccp_alpha=param_val,
                                           max_depth=max_depth_val,
                                           random_state=42)
        else:
            param_val = rf_base_factor
            n_estimators_val = n_estimators if n_estimators is not None else 100
            max_features_val = max(1, int(param_val * X_train_full.shape[1]))
            model = RandomForestClassifier(n_estimators=n_estimators_val,
                                           max_features=max_features_val,
                                           random_state=42, oob_score=True)
        model.fit(X_train_full, y_train_full)
        return (model, X_train_full, y_train_full, X_test_full, y_test_full, X_train_vis, X_test_vis)

    def plot_decision_boundary(model, X_origin, X_vis):
        Z_pred = model.predict(X_origin)
        x_min, x_max = X_vis['TSNE1'].min() - 1, X_vis['TSNE1'].max() + 1
        y_min, y_max = X_vis['TSNE2'].min() - 1, X_vis['TSNE2'].max() + 1
        h = 200
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, h), np.linspace(y_min, y_max, h))
        background_model = KNeighborsClassifier(n_neighbors=1).fit(X_vis, Z_pred)
        voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape((h, h))
        return (xx, yy, voronoiBackground)

    def plot_accuracy_curve(x_axis_choice, model_type, split, max_depth, n_estimators):
        acc_train_list, acc_test_list = [], []
        if x_axis_choice == "dt_ccp_alpha":
            param_vals = np.linspace(0, 0.2, 21)
            for p in param_vals:
                m, X_tr, y_tr, X_te, y_te, _, _ = train_model(model_type, split, p, None, max_depth, None)
                acc_train_list.append(accuracy_score(y_tr, m.predict(X_tr)))
                acc_test_list.append(accuracy_score(y_te, m.predict(X_te)))
        elif x_axis_choice == "max_depth":
            if model_type == "dt":
                param_vals = np.arange(1, 21)
                for p in param_vals:
                    m, X_tr, y_tr, X_te, y_te, _, _ = train_model(model_type, split, input.dt_ccp_alpha(), None, p, None)
                    acc_train_list.append(accuracy_score(y_tr, m.predict(X_tr)))
                    acc_test_list.append(accuracy_score(y_te, m.predict(X_te)))
            else:
                param_vals = []
        elif x_axis_choice == "rf_base_factor":
            if model_type == "rf":
                param_vals = np.linspace(0, 0.2, 21)
                for p in param_vals:
                    m, X_tr, y_tr, X_te, y_te, _, _ = train_model(model_type, split, None, p, max_depth, n_estimators)
                    acc_train_list.append(accuracy_score(y_tr, m.predict(X_tr)))
                    acc_test_list.append(accuracy_score(y_te, m.predict(X_te)))
            else:
                param_vals = []
        elif x_axis_choice == "n_estimators":
            if model_type == "rf":
                param_vals = np.arange(50, 501, 10)
                for p in param_vals:
                    m, X_tr, y_tr, X_te, y_te, _, _ = train_model(model_type, split, None, input.rf_base_factor(), max_depth, p)
                    acc_train_list.append(accuracy_score(y_tr, m.predict(X_tr)))
                    acc_test_list.append(accuracy_score(y_te, m.predict(X_te)))
            else:
                param_vals = []
        else:
            param_vals = []
        return param_vals, acc_train_list, acc_test_list

    @output
    @render.plot
    def model_plot():
        if input.model_type() == "dt":
            dt_val = input.dt_ccp_alpha()
            rf_val = None
        else:
            dt_val = None
            rf_val = input.rf_base_factor()
        model_pack = train_model(input.model_type(), input.use_split(), dt_val, rf_val,
                                 input.max_depth() if input.model_type() == "dt" else None,
                                 input.n_estimators() if input.model_type() == "rf" else None)
        model_inst, X_train_full, y_train_full, X_test_full, y_test_full, X_train_vis, X_test_vis = model_pack
        fig = plt.figure(figsize=(16,14))
        ax1 = fig.add_subplot(2,1,1)
        if input.use_split():
            train_class0 = X_train_vis[y_train_full==0]
            train_class1 = X_train_vis[y_train_full==1]
            test_class0 = X_test_vis[y_test_full==0]
            test_class1 = X_test_vis[y_test_full==1]
            xx, yy, Z = plot_decision_boundary(model_inst, X_train_full, X_train_vis)
            ax1.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
            ax1.scatter(train_class0['TSNE1'], train_class0['TSNE2'], c='skyblue', label="Class 0 - Train", edgecolor='k')
            ax1.scatter(train_class1['TSNE1'], train_class1['TSNE2'], c='orange', label="Class 1 - Train", edgecolor='k')
            ax1.scatter(test_class0['TSNE1'], test_class0['TSNE2'], c='navy', marker='^', label="Class 0 - Test", edgecolor='k')
            ax1.scatter(test_class1['TSNE1'], test_class1['TSNE2'], c='yellow', marker='^', label="Class 1 - Test", edgecolor='k')
            ax1.set_title("Decision Boundary & Data Scatter Plot")
        else:
            xx, yy, Z = plot_decision_boundary(model_inst, X_train_full, X_train_vis)
            ax1.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
            class0 = X_train_vis[y_train_full==0]
            class1 = X_train_vis[y_train_full==1]
            ax1.scatter(class0['TSNE1'], class0['TSNE2'], c='blue', label="Class 0", edgecolor='k')
            ax1.scatter(class1['TSNE1'], class1['TSNE2'], c='red', label="Class 1", edgecolor='k')
            ax1.set_title("Decision Boundary & Data Scatter Plot")
        ax1.set_xlabel("TSNE1")
        ax1.set_ylabel("TSNE2")
        ax1.legend()

        ax2 = fig.add_subplot(2,1,2)
        xaxis_choice = input.xaxis_choice()
        param_vals, acc_train_list, acc_test_list = plot_accuracy_curve(xaxis_choice,
                                                                         input.model_type(),
                                                                         input.use_split(),
                                                                         input.max_depth() if input.model_type() == "dt" else None,
                                                                         input.n_estimators() if input.model_type() == "rf" else None)
        ax2.plot(param_vals, acc_train_list, label="Train Accuracy", color='green', marker='o')
        ax2.plot(param_vals, acc_test_list, label="Test Accuracy", color='purple', marker='o')
        if xaxis_choice == "dt_ccp_alpha":
            current_val = input.dt_ccp_alpha()
        elif xaxis_choice == "max_depth":
            current_val = input.max_depth()
        elif xaxis_choice == "rf_base_factor":
            current_val = input.rf_base_factor()
        elif xaxis_choice == "n_estimators":
            current_val = input.n_estimators()
        idx = (np.abs(param_vals - current_val)).argmin()
        ax2.plot(param_vals[idx], acc_train_list[idx], marker='o', markersize=12, color='darkgreen')
        ax2.plot(param_vals[idx], acc_test_list[idx], marker='o', markersize=12, color='darkviolet')
        ax2.set_xlabel(f"{xaxis_choice}")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy vs " + f"{xaxis_choice}")
        ax2.legend()
        plt.subplots_adjust(hspace=0.5, bottom=0.15)
        mplcursors.cursor(ax2.get_lines(), hover=True)
        return fig

    @output
    @render.ui
    def model_eval():
        if input.model_type() == "dt":
            dt_val = input.dt_ccp_alpha()
            rf_val = None
        else:
            dt_val = None
            rf_val = input.rf_base_factor()
        model_pack = train_model(input.model_type(), input.use_split(), dt_val, rf_val,
                                 input.max_depth() if input.model_type() == "dt" else None,
                                 input.n_estimators() if input.model_type() == "rf" else None)
        if input.use_split():
            y_train_pred = model_pack[0].predict(model_pack[1])
            y_test_pred = model_pack[0].predict(model_pack[3])
            y_train_prob = model_pack[0].predict_proba(model_pack[1])[:, 1]
            y_test_prob = model_pack[0].predict_proba(model_pack[3])[:, 1]
            html_train = build_confusion_metrics_table(model_pack[2], y_train_pred, y_train_prob)
            html_test = build_confusion_metrics_table(model_pack[4], y_test_pred, y_test_prob)
            combined = html_train + "<br><br>" + html_test
            return ui.HTML(combined)
        else:
            y_pred = model_pack[0].predict(model_pack[1])
            y_prob = model_pack[0].predict_proba(model_pack[1])[:, 1]
            html_full = build_confusion_metrics_table(model_pack[2], y_pred, y_prob)
            return ui.HTML(html_full)

    @output
    @render.text
    def model_info():
        return ""

import nest_asyncio
nest_asyncio.apply()

if __name__ == "__main__":
    app = App(app_ui, server)
    app.run()