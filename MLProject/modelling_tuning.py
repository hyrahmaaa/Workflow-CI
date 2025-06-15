# Workflow-CI/MLProject/modelling.py (file ini berisi logika dari modelling_tuning.py Anda)

# -*- coding: utf-8 -*-
"""modelling_tuning.py (Adapted from modelling_tuning.ipynb for MLflow Project CI)"""

import pandas as pd
import numpy as np
import mlflow
import dagshub
import joblib  
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

PROCESSED_DATA_FOLDER_NAME = 'telco_churn_preprocessing'
PATH_TO_PROCESSED_DATA = os.path.join('.', PROCESSED_DATA_FOLDER_NAME)

DAGSHUB_USERNAME = "hyrahmaaa" 
DAGSHUB_REPO_NAME = "Workflow-CI" 

MLFLOW_TRACKING_URI = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"


def load_processed_data(path):
    """
    Memuat data training dan testing yang sudah diproses.
    """
    print(f"Memuat data yang diproses dari: {path}")
    if not os.path.exists(path):
        print(f"Error: Direktori '{path}' tidak ditemukan.")
        print("Pastikan Anda telah menjalankan langkah preprocessing dan menyimpan data di lokasi ini.")
        return None, None, None, None

    try:
        X_train = pd.read_csv(os.path.join(path, 'X_train.csv'))
        X_test = pd.read_csv(os.path.join(path, 'X_test.csv'))
        y_train = pd.read_csv(os.path.join(path, 'y_train.csv')).squeeze()
        y_test = pd.read_csv(os.path.join(path, 'y_test.csv')).squeeze()
        print("Data yang diproses berhasil dimuat.")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        print(f"Error: File tidak ditemukan di '{path}'. Detail: {e}")
        print("Pastikan semua file (X_train.csv, X_test.csv, y_train.csv, y_test.csv) ada di direktori yang ditentukan.")
        return None, None, None, None

if __name__ == "__main__":
    print("--- Memulai Hyperparameter Tuning dan Manual Logging dengan MLflow ---")

    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME) 

    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ.get("MLFLOW_TRACKING_PASSWORD", "") 


    # Muat data yang sudah diproses
    X_train, X_test, y_train, y_test = load_processed_data(PATH_TO_PROCESSED_DATA)

    # Check if data was loaded successfully
    if X_train is None:
        print("\n--- Tuning Model Dibatalkan karena data tidak dapat dimuat. ---")
    else:
        # Definisikan model dasar
        model_base = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)

        # Definisikan grid parameter untuk tuning
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2']
        }

        # Inisialisasi GridSearchCV
        grid_search = GridSearchCV(estimator=model_base, param_grid=param_grid, 
                                   cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

        print("Memulai pencarian hyperparameter...")
        grid_search.fit(X_train, y_train)
        print("Pencarian hyperparameter selesai.")

        # Dapatkan model terbaik
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_ # Ini adalah mean cross-validated score dari best_estimator

        print(f"\nModel Terbaik: {best_model}")
        print(f"Parameter Terbaik: {best_params}")
        print(f"ROC AUC Terbaik (Cross-validation): {best_score:.4f}")

        # Lakukan prediksi dan evaluasi pada test set dengan model terbaik
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        final_accuracy = accuracy_score(y_test, y_pred)
        final_precision = precision_score(y_test, y_pred)
        final_recall = recall_score(y_test, y_pred)
        final_f1 = f1_score(y_test, y_pred)
        final_roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"Final Accuracy (Test Set): {final_accuracy:.4f}")
        print(f"Final Precision (Test Set): {final_precision:.4f}")
        print(f"Final Recall (Test Set): {final_recall:.4f}")
        print(f"Final F1-Score (Test Set): {final_f1:.4f}")
        print(f"Final ROC AUC (Test Set): {final_roc_auc:.4f}")


        # --- MANUAL LOGGING DENGAN MLFLOW ---
        with mlflow.start_run(run_name="Tuned_Logistic_Regression_Best_Run"):
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


            mlflow_local_artifact_path_uri = mlflow.active_run().info.artifact_uri
            mlflow_local_artifact_path = mlflow_local_artifact_path_uri.replace("file://", "")

            model_artifact_dir_local = os.path.join(mlflow_local_artifact_path, "model")
            os.makedirs(model_artifact_dir_local, exist_ok=True)

            model_path_local_final = os.path.join(model_artifact_dir_local, "best_logistic_regression_model.pkl")
            joblib.dump(best_model, model_path_local_final)
            print(f"Model saved locally to: {model_path_local_final}") 

            - name: Create Artifacts Directory
              run: |
                mkdir -p artifacts/model
                ls -la artifacts/
        
            dagshub.upload_artifact(
                repo_url=f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.git",
                file_path=model_path_local_final, 
                path_on_dagshub="model/best_logistic_regression_model.pkl", 
                message=f"Model artifact from CI retraining {mlflow.active_run().info.run_id}"
            )
            print(f"Model artifact uploaded directly to DagsHub via dagshub.upload_artifact")

            print("\n--- Tuning Model Selesai. Hasil dicatat ke MLflow. ---")
            print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

print("\n--- Proses Tuning dan Logging Selesai. Periksa MLflow UI! ---")
