# MLProject/predict_model.py
import mlflow
import pandas as pd
import numpy as np
import os

def load_and_predict(input_data_df):
    """
    Memuat model MLflow yang sudah dilatih dan melakukan prediksi.

    Args:
        input_data_df (pd.DataFrame): DataFrame berisi data input untuk prediksi.
                                      Kolom harus sesuai dengan fitur yang diharapkan model.

    Returns:
        np.ndarray: Hasil prediksi dari model.
    """
    MODEL_PATH_IN_CONTAINER = "./best_logistic_regression_model_artifact" 

    if not os.path.exists(MODEL_PATH_IN_CONTAINER):
        print(f"Error: Model not found at {MODEL_PATH_IN_CONTAINER}")
        print("Please ensure the model artifact folder is copied correctly into the Docker image.")
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH_IN_CONTAINER}")


    print(f"Loading model from: {MODEL_PATH_IN_CONTAINER}")
    loaded_model = mlflow.pyfunc.load_model(MODEL_PATH_IN_CONTAINER)
    print("Model loaded successfully.")

    prediction = loaded_model.predict(input_data_df)
    return prediction

if __name__ == "__main__":
    print("--- Running prediction script (inside Docker/locally) ---")

    num_features = 30 
    dummy_input_array = np.random.rand(1, num_features)
    dummy_columns = [f'feature_{i+1}' for i in range(num_features)] 

    dummy_df = pd.DataFrame(dummy_input_array, columns=dummy_columns)

    print("\nDummy Input Data for Prediction:")
    print(dummy_df)

    try:
        predictions = load_and_predict(dummy_df)
        print("\nPrediction Result:")
        print(predictions)

        if len(predictions) > 0 and predictions[0] in [0, 1]:
            print(f"Prediction (0=No Churn, 1=Churn): {predictions[0]}")

    except FileNotFoundError as e:
        print(f"Error during prediction: {e}")
        print("Please check if the model artifact folder 'best_logistic_regression_model_artifact' is copied correctly into the Docker image at the specified path.")
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")

    print("\n--- Prediction script finished ---")
