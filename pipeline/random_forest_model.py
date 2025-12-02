import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib
import datetime as dt

LOG_PATH = Path("logs/pipeline.log")


def log(msg):
    LOG_PATH.parent.mkdir(exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ----------------------------------------------------------
# ENTRENAMIENTO RANDOM FOREST CON SPLIT TEMPORAL
# ----------------------------------------------------------
def run_rf_training():
    data_path = Path("output/master_table.csv")
    log("=== INICIO ENTRENAMIENTO RANDOM FOREST ===")

    df = pd.read_csv(data_path)
    df["last_review_date"] = pd.to_datetime(df["last_review_date"], errors="coerce")

    # SPLITS
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

    target_col = "cust_review_mean"

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    rf_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    param_grid = {
        "model__n_estimators": [200, 300],
        "model__max_depth": [10, 20]
    }

    search = GridSearchCV(
        rf_pipe,
        param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        refit=True,
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    log(f"BEST PARAMS RF: {search.best_params_}")

    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    log(f"MSE TEST={mse:.4f}")
    log(f"R2 TEST={r2:.4f}")

    joblib.dump(best_model, "best_model_rf.pkl")
    df_holdout.to_csv("holdout_segment.csv", index=False)

    log("Modelo RF guardado como best_model_rf.pkl")
    log("=== FIN ENTRENAMIENTO RANDOM FOREST ===")

    return best_model, df_holdout
