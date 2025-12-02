import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

import datetime as dt

LOG_PATH = Path("logs/pipeline.log")


def log(msg):
    LOG_PATH.parent.mkdir(exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ----------------------------------------------------------
# ENTRENAMIENTO XGBOOST CON SPLIT TEMPORAL
# ----------------------------------------------------------
def run_xgb_training():
    data_path = Path("output/master_table.csv")
    log("=== INICIO ENTRENAMIENTO XGBOOST ===")

    df = pd.read_csv(data_path)

    # ------------------------------------------------------
    # Convertir fecha
    # ------------------------------------------------------
    df["last_review_date"] = pd.to_datetime(df["last_review_date"], errors="coerce")

    # ------------------------------------------------------
    # DEFINIR SPLITS
    # ------------------------------------------------------
    train_start = dt.datetime(2016, 10, 1)
    train_end = dt.datetime(2018, 3, 31)

    test_start = dt.datetime(2018, 4, 1)
    test_end = dt.datetime(2018, 9, 30)

    holdout_start = dt.datetime(2018, 10, 1)
    holdout_end = dt.datetime(2018, 12, 31)

    df_train = df[(df["last_review_date"] >= train_start) & (df["last_review_date"] <= train_end)].copy()
    df_test = df[(df["last_review_date"] >= test_start) & (df["last_review_date"] <= test_end)].copy()
    df_holdout = df[(df["last_review_date"] >= holdout_start) & (df["last_review_date"] <= holdout_end)].copy()

    log(f"TRAIN: {df_train.shape}")
    log(f"TEST: {df_test.shape}")
    log(f"HOLDOUT: {df_holdout.shape}")

    # ------------------------------------------------------
    # Preparar X / y
    # ------------------------------------------------------
    target_col = "cust_review_mean"

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    # ------------------------------------------------------
    # COLUMNAS
    # ------------------------------------------------------
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    # ------------------------------------------------------
    # PREPROCESSING
    # ------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    # ------------------------------------------------------
    # XGBOOST PIPELINE
    # ------------------------------------------------------
    xgb_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", XGBRegressor(
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror"
        ))
    ])

    # ------------------------------------------------------
    # GRID SEARCH
    # ------------------------------------------------------
    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [4, 6],
        "model__learning_rate": [0.05, 0.1]
    }

    search = GridSearchCV(
        xgb_pipe,
        param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        refit=True,
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    log(f"BEST PARAMS XGB: {search.best_params_}")

    # ------------------------------------------------------
    # EVALUACIÓN
    # ------------------------------------------------------
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    log(f"MSE TEST={mse:.4f}")
    log(f"R2 TEST={r2:.4f}")

    # ------------------------------------------------------
    # GUARDAR MODELO Y HOLDOUT
    # ------------------------------------------------------
    joblib.dump(best_model, "best_model_xgb.pkl")
    df_holdout.to_csv("holdout_segment.csv", index=False)

    log("Modelo XGB guardado como best_model_xgb.pkl")
    log("=== FIN ENTRENAMIENTO XGBOOST ===")

    return best_model, df_holdout
