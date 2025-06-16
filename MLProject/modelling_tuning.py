# MLProject/modelling_tuning.py

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import joblib 
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

# Import kelas ChurnPredictor dari inference.py
# Kita asumsikan inference.py ada di direktori yang sama dengan modelling_tuning.py
# Pastikan inference.py sudah bersih seperti yang terakhir kita diskusikan (bagian __main__ dikomentari)
from inference import ChurnPredictor 

PROCESSED_DATA_FOLDER_NAME = 'telco_churn_preprocessing'
PATH_TO_PROCESSED_DATA = os.path.join('.', PROCESSED_DATA_FOLDER_NAME) 

DAGSHUB_USERNAME = "hyrahmaaa" 
DAGSHUB_REPO_NAME = "CI-Workflow" 

def load_processed_data(path):
    """
    Memuat data training dan testing yang sudah diproses.
    """
    print(f"Memuat data yang diproses dari: {path}")
    # Perbaiki path absolut agar berfungsi di CI/CD dan lokal
    # Menggunakan os.path.abspath untuk mendapatkan path saat ini
    base_dir = os.path.abspath(os.path.dirname(__file__))
    absolute_path = os.path.join(base_dir, path)
    
    if not os.path.exists(absolute_path):
        print(f"Error: Direktori '{absolute_path}' tidak ditemukan.")
        print("Pastikan Anda telah menjalankan langkah preprocessing dan menyimpan data di lokasi ini.")
        return None, None, None, None

    try:
        X_train = pd.read_csv(os.path.join(absolute_path, 'X_train.csv'))
        X_test = pd.read_csv(os.path.join(absolute_path, 'X_test.csv'))
        y_train = pd.read_csv(os.path.join(absolute_path, 'y_train.csv')).squeeze()
        y_test = pd.read_csv(os.path.join(absolute_path, 'y_test.csv')).squeeze()
        print("Data yang diproses berhasil dimuat.")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        print(f"Error: File tidak ditemukan di '{absolute_path}'. Detail: {e}")
        print("Pastikan semua file (X_train.csv, X_test.csv, y_train.csv, y_test.csv) ada di direktori yang ditentukan.")
        return None, None, None, None

if __name__ == "__main__":
    print("--- Memulai Hyperparameter Tuning dan Logging dengan MLflow ---")

    X_train, X_test, y_train, y_test = load_processed_data(PATH_TO_PROCESSED_DATA)

    if X_train is None:
        print("\n--- Tuning Model Dibatalkan karena data tidak dapat dimuat. ---")
    else:
        model_base = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2']
        }

        grid_search = GridSearchCV(estimator=model_base, param_grid=param_grid, 
                                   cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

        print("Memulai pencarian hyperparameter...")
        grid_search.fit(X_train, y_train)
        print("Pencarian hyperparameter selesai.")

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_ 
        
        print(f"\nModel Terbaik: {best_model}")
        print(f"Parameter Terbaik: {best_params}")
        print(f"ROC AUC Terbaik (Cross-validation): {best_score:.4f}")

        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        final_accuracy = accuracy_score(y_test, y_pred)
        final_precision = precision_score(y_test, y_pred)
        final_recall = recall_score(y_test, y_pred)
        final_f1 = f1_score(y_test, final_recall, final_precision) # Perbaiki ini
        final_roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"Final Accuracy (Test Set): {final_accuracy:.4f}")
        print(f"Final Precision (Test Set): {final_precision:.4f}")
        print(f"Final Recall (Test Set): {final_recall:.4f}")
        print(f"Final F1-Score (Test Set): {final_f1:.4f}")
        print(f"Final ROC AUC (Test Set): {final_roc_auc:.4f}")

        with mlflow.start_run(run_name="Tuned_Logistic_Regression_Best_Run") as run:
            print(f"MLflow run started with ID: {run.info.run_id}") 

            mlflow.log_params(best_params)
            mlflow.log_param("model_type", type(best_model).__name__)
            mlflow.log_param("cv_strategy", "GridSearchCV")
            mlflow.log_param("cv_folds", grid_search.cv) 
            
            mlflow.log_metric("test_accuracy", final_accuracy)
            mlflow.log_metric("test_precision", final_precision)
            mlflow.log_metric("test_recall", final_recall)
            mlflow.log_metric("test_f1_score", final_f1)
            mlflow.log_metric("test_roc_auc", final_roc_auc)
            mlflow.log_metric("best_cv_roc_auc", best_score)

            # --- Perubahan Kritis di Sini ---
            # Mengemas model dengan custom PythonModel (ChurnPredictor dari inference.py)
            # dan menyertakan model terlatih sebagai artefak
            mlflow.pyfunc.log_model(
                artifact_path="best_logistic_regression_model_artifact",
                python_model=ChurnPredictor(),
                artifacts={"model_path": best_model}, # Menyertakan model terlatih sebagai artefak yang akan diakses ChurnPredictor
                # Menambahkan signature untuk input dan output model
                input_example=X_train.head(1), 
                signature=mlflow.models.signature.infer_signature(X_train, y_pred)
            )
            print("mlflow.pyfunc.log_model called successfully with custom PythonModel. Model should be logged to MLflow artifacts.") # DEBUGGING

        print("\n--- Tuning Model Selesai. Hasil dicatat ke MLflow. ---")

print("\n--- Proses Tuning dan Logging Selesai. Periksa MLflow UI! ---")
