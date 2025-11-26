import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import joblib

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception as e:
    HAS_XGB = False


# ----------------------------------------------------------
# LOGGER
# ----------------------------------------------------------
def setup_logger():
    logger = logging.getLogger("model_training")
    logger.setLevel(logging.INFO)

    Path("logs").mkdir(exist_ok=True)
    fh = logging.FileHandler("logs/pipeline.log", mode="w", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)

    if not logger.handlers:
        logger.addHandler(fh)

    return logger


# ----------------------------------------------------------
# TRAINING PIPELINE
# ----------------------------------------------------------
def run_training_pipeline():
    data_path="output/master_table.csv"
    logger = setup_logger()
    logger.info("=== INICIO ENTRENAMIENTO DE MODELOS ===")
    logger.info(f"Leyendo archivo: {data_path}")

    data_path = Path(data_path)

    # ------------------------------------------------------
    # CARGA DE CSV
    # ------------------------------------------------------
    try:
        df = pd.read_csv(data_path)
        logger.info(f"CSV cargado correctamente. Shape: {df.shape}")
    except Exception as e:
        logger.error(f"ERROR cargando CSV: {e}")
        return None, None

    # ------------------------------------------------------
    # FILTRO unknown
    # ------------------------------------------------------
    mask_unknown = df["last_review_date"].eq("unknown")
    logger.info(f"Filas con unknown: {mask_unknown.sum()}")

    df = df.loc[~mask_unknown].copy()
    logger.info(f"Shape tras filtro unknown: {df.shape}")

    # ------------------------------------------------------
    # PREPROCESO FECHA
    # ------------------------------------------------------
    df["last_review_date"] = pd.to_datetime(df["last_review_date"], errors="coerce")
    df["last_review_year"] = df["last_review_date"].dt.year
    df["last_review_month"] = df["last_review_date"].dt.month

    # ------------------------------------------------------
    # DROP COLUMNAS
    # ------------------------------------------------------
    cols_drop = [
        "last_review_date",
        "customer_id",
        "log_cust_review_mean",
        "cust_review_std",
        "cust_num_good_reviews",
        "cust_num_bad_reviews"
    ]

    df = df.drop(columns=[c for c in cols_drop if c in df.columns], errors="ignore")

    # ------------------------------------------------------
    # SPLIT X / y
    # ------------------------------------------------------
    target_col = "cust_review_mean"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ------------------------------------------------------
    # TIPOS DE VARIABLES
    # ------------------------------------------------------
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    logger.info(f"Columnas numéricas: {numeric_cols}")
    logger.info(f"Columnas categóricas: {categorical_cols}")

    # ------------------------------------------------------
    # PREPROCESSOR
    # ------------------------------------------------------
    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    # ------------------------------------------------------
    # TRAIN TEST SPLIT
    # ------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    results = []
    best_model = None
    best_model_name = None
    best_mse = np.inf

    # ----------------------------------------------------------
    # 1. LINEAR REGRESSION
    # ----------------------------------------------------------
    lr_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", LinearRegression())
    ])

    lr_grid = {
        "model__fit_intercept": [True, False],
        "model__n_jobs": [None, -1]
    }

    lr_search = GridSearchCV(
        lr_pipe, lr_grid, cv=5,
        scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
        refit="mse",
        n_jobs=-1
    )
    lr_search.fit(X_train, y_train)

    idx = lr_search.best_index_
    lr_mse = -lr_search.cv_results_["mean_test_mse"][idx]
    lr_r2 = lr_search.cv_results_["mean_test_r2"][idx]

    logger.info(f"LR mejores params: {lr_search.best_params_}")
    logger.info(f"LR CV MSE={lr_mse:.4f}, R2={lr_r2:.4f}")

    results.append({
        "model": "LinearRegression",
        "mse": lr_mse,
        "r2": lr_r2,
        "best_params": lr_search.best_params_
    })

    if lr_mse < best_mse:
        best_mse = lr_mse
        best_model = lr_search.best_estimator_
        best_model_name = "LinearRegression"

    # ----------------------------------------------------------
    # 2. RANDOM FOREST
    # ----------------------------------------------------------
    rf_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    rf_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20]
    }

    rf_search = GridSearchCV(
        rf_pipe, rf_grid, cv=5,
        scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
        refit="mse",
        n_jobs=-1
    )
    rf_search.fit(X_train, y_train)

    idx = rf_search.best_index_
    rf_mse = -rf_search.cv_results_["mean_test_mse"][idx]
    rf_r2 = rf_search.cv_results_["mean_test_r2"][idx]

    logger.info(f"RF mejores params: {rf_search.best_params_}")
    logger.info(f"RF CV MSE={rf_mse:.4f}, R2={rf_r2:.4f}")

    results.append({
        "model": "RandomForest",
        "mse": rf_mse,
        "r2": rf_r2,
        "best_params": rf_search.best_params_
    })

    if rf_mse < best_mse:
        best_mse = rf_mse
        best_model = rf_search.best_estimator_
        best_model_name = "RandomForest"

    # ----------------------------------------------------------
    # 3. GRADIENT BOOSTING
    # ----------------------------------------------------------
    gb_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", GradientBoostingRegressor(random_state=42))
    ])

    gb_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.05, 0.1, 0.2]
    }

    gb_search = GridSearchCV(
        gb_pipe, gb_grid, cv=5,
        scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
        refit="mse",
        n_jobs=-1
    )
    gb_search.fit(X_train, y_train)

    idx = gb_search.best_index_
    gb_mse = -gb_search.cv_results_["mean_test_mse"][idx]
    gb_r2 = gb_search.cv_results_["mean_test_r2"][idx]

    logger.info(f"GB mejores params: {gb_search.best_params_}")
    logger.info(f"GB CV MSE={gb_mse:.4f}, R2={gb_r2:.4f}")

    results.append({
        "model": "GradientBoosting",
        "mse": gb_mse,
        "r2": gb_r2,
        "best_params": gb_search.best_params_
    })

    if gb_mse < best_mse:
        best_mse = gb_mse
        best_model = gb_search.best_estimator_
        best_model_name = "GradientBoosting"

    # ----------------------------------------------------------
    # 4. XGBOOST (solo si disponible)
    # ----------------------------------------------------------
    if HAS_XGB:
        xgb_pipe = Pipeline([
            ("prep", preprocessor),
            ("model", XGBRegressor(
                random_state=42,
                n_jobs=-1,
                objective="reg:squarederror"
            ))
        ])

        xgb_grid = {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [4, 6, 8],
            "model__learning_rate": [0.05, 0.1, 0.2]
        }

        xgb_search = GridSearchCV(
            xgb_pipe, xgb_grid, cv=5,
            scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
            refit="mse",
            n_jobs=-1
        )
        xgb_search.fit(X_train, y_train)

        idx = xgb_search.best_index_
        xgb_mse = -xgb_search.cv_results_["mean_test_mse"][idx]
        xgb_r2 = xgb_search.cv_results_["mean_test_r2"][idx]

        logger.info(f"XGB mejores params: {xgb_search.best_params_}")
        logger.info(f"XGB CV MSE={xgb_mse:.4f}, R2={xgb_r2:.4f}")

        results.append({
            "model": "XGBoost",
            "mse": xgb_mse,
            "r2": xgb_r2,
            "best_params": xgb_search.best_params_
        })

        if xgb_mse < best_mse:
            best_mse = xgb_mse
            best_model = xgb_search.best_estimator_
            best_model_name = "XGBoost"

    # ----------------------------------------------------------
    # EVALUACIÓN EN TEST
    # ----------------------------------------------------------
    y_pred = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    logger.info(f"MEJOR MODELO: {best_model_name}")
    logger.info(f"MSE CV={best_mse:.4f}")
    logger.info(f"MSE TEST={test_mse:.4f}")
    logger.info(f"R2 TEST={test_r2:.4f}")

    # ----------------------------------------------------------
    # GUARDAR MODELO
    # ----------------------------------------------------------
    joblib.dump(best_model, "best_model.pkl")
    logger.info("Modelo guardado: best_model.pkl")

    results_df = pd.DataFrame(results)

    logger.info("=== FIN ENTRENAMIENTO ===")

    return best_model, results_df
