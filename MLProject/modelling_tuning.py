# Workflow-CI/MLProject/modelling_tuning.py

# -*- coding: utf-8 -*-
"""modelling_tuning.py (Adapted from modelling_tuning.ipynb for MLflow Project CI)"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import joblib # joblib dipertahankan jika Anda ingin menyimpan secara lokal di luar MLflow, tapi untuk log_model tidak perlu
import os

PROCESSED_DATA_FOLDER_NAME = 'telco_churn_preprocessing'
# PATH_TO_PROCESSED_DATA harus relatif terhadap lokasi script ini
PATH_TO_PROCESSED_DATA = os.path.join('.', PROCESSED_DATA_FOLDER_NAME) 

DAGSHUB_USERNAME = "hyrahmaaa" 
DAGSHUB_REPO_NAME = "Workflow-CI" 

# MLFLOW_TRACKING_URI akan diatur oleh GitHub Actions, tidak perlu di-hardcode di sini
# MLFLOW_TRACKING_URI = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"


def load_processed_data(path):
    """
    Memuat data training dan testing yang sudah diproses.
    """
    print(f"Memuat data yang diproses dari: {path}")
    # Perbaiki path relatif untuk preprocessing data
    # Karena modelling_tuning.py ada di MLProject/, dan telco_churn_preprocessing juga di MLProject/
    # maka path harus langsung ke telco_churn_preprocessing
    absolute_path = os.path.join(os.path.dirname(__file__), path)
    
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

    # Inisialisasi Dagshub di sini. Penting agar MLflow Tracking URI bisa diatur
    # Hati-hati dengan mlflow_tracking=True jika pernah menyebabkan TypeError
    # Jika MLFLOW_TRACKING_URI sudah disetel di GHA, ini mungkin hanya untuk credential
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME) 

    # MLFLOW_TRACKING_URI, USERNAME, PASSWORD sudah disetel di GitHub Actions
    # TIDAK perlu di-set ulang di dalam script Python ini
    # os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    # os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    # os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ.get("MLFLOW_TRACKING_PASSWORD", "") 

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


        # --- LOGGING DENGAN MLFLOW ---
        # MLflow akan secara otomatis menggunakan MLFLOW_TRACKING_URI dari env
        with mlflow.start_run(run_name="Tuned_Logistic_Regression_Best_Run") as run:
            print(f"MLflow run started with ID: {run.info.run_id}") # DEBUGGING

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

            # Ini adalah cara standar untuk melog model dengan MLflow
            # MLflow akan menyimpan model ini ke folder artifacts di dalam run
            # dan mengunggahnya ke Tracking URI (DagsHub)
            mlflow.sklearn.log_model(best_model, "best_logistic_regression_model_artifact")
            print("mlflow.sklearn.log_model called successfully. Model should be logged to MLflow artifacts.") # DEBUGGING

        print("\n--- Tuning Model Selesai. Hasil dicatat ke MLflow. ---")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

print("\n--- Proses Tuning dan Logging Selesai. Periksa MLflow UI! ---")
