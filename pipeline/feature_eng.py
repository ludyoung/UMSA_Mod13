import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from pipeline.utils import log

def generar_features(dfs: dict) -> pd.DataFrame:
    """
    Genera features avanzados para análisis de satisfacción del cliente.
    Incluye features por producto, vendedor, pagos, geográficos y
    temporales (mensual, trimestral y semestral).
    Realiza selección de features con RandomForest.
    """

    log("=== INICIO Feature Engineering Avanzado ===")

    # -------------------------------
    # Cargar tablas base
    # -------------------------------
    log("Cargando tablas base...")
    orders = dfs["orders"]
    order_items = dfs["order_items"]
    reviews = dfs["reviews"]
    products = dfs["products"]
    sellers = dfs["sellers"]
    payments = dfs["payments"]
    customers = dfs["customers"]
    geo = dfs["geolocation"]
    cat_trans = dfs["category_translation"]

    # -------------------------------
    # Merge maestro
    # -------------------------------
    log("Realizando merge global...")
    df = (
        orders.merge(order_items, on="order_id", how="left")
        .merge(reviews, on="order_id", how="left")
        .merge(products, on="product_id", how="left")
        .merge(sellers, on="seller_id", how="left")
        .merge(payments, on="order_id", how="left")
        .merge(customers, on="customer_id", how="left")
    )

    log(f"Dataset base creado con shape {df.shape}")

    # -------------------------------
    # Fechas
    # -------------------------------
    log("Procesando variables temporales...")
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df["delivery_time_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
    df["delay_vs_estimated"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days
    df["approval_delay"] = (df["order_approved_at"] - df["order_purchase_timestamp"]).dt.total_seconds() / 3600
    df["carrier_delay"] = (df["order_delivered_carrier_date"] - df["order_approved_at"]).dt.total_seconds() / 3600
    df["was_late"] = (df["delay_vs_estimated"] > 0).astype(int)

    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_day"] = df["order_purchase_timestamp"].dt.day
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.weekday
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour

    # -------------------------------
    # Features por producto
    # -------------------------------
    log("Generando features por producto...")
    product_group = df.groupby("product_id").agg(
        prod_price_mean=("price", "mean"),
        prod_price_std=("price", "std"),
        prod_weight_mean=("product_weight_g", "mean"),
        prod_weight_std=("product_weight_g", "std"),
        prod_volume=("product_length_cm", "mean"),
        prod_is_heavy=("product_weight_g", lambda x: (x > x.mean()).mean()),
        prod_review_mean=("review_score", "mean"),
        prod_review_std=("review_score", "std"),
    )
    df = df.merge(product_group, on="product_id", how="left")

    # -------------------------------
    # Features del vendedor
    # -------------------------------
    log("Generando features del vendedor...")
    seller_group = df.groupby("seller_id").agg(
        seller_num_products=("product_id", "nunique"),
        seller_total_orders=("order_id", "count"),
        seller_avg_price=("price", "mean"),
        seller_review_mean=("review_score", "mean"),
    )
    df = df.merge(seller_group, on="seller_id", how="left")

    # -------------------------------
    # Features de pagos
    # -------------------------------
    log("Generando features de pago...")
    payment_group = df.groupby("customer_id").agg(
        pay_total=("payment_value", "sum"),
        pay_installments_mean=("payment_installments", "mean"),
        pay_installments_max=("payment_installments", "max"),
        pay_num_methods=("payment_type", "nunique"),
    )
    df = df.merge(payment_group, on="customer_id", how="left")

    # -------------------------------
    # Features por cliente
    # -------------------------------
    log("Generando features agregados por cliente...")
    customer_features = df.groupby("customer_id").agg(
        cust_total_orders=("order_id", "count"),
        cust_total_spent=("price", "sum"),
        cust_avg_price=("price", "mean"),
        cust_avg_freight=("freight_value", "mean"),
        cust_num_products=("product_id", "nunique"),
        cust_review_mean=("review_score", "mean"),
        cust_review_std=("review_score", "std"),
        cust_num_bad_reviews=("review_score", lambda x: (x <= 2).sum()),
        cust_num_good_reviews=("review_score", lambda x: (x >= 4).sum()),
        cust_avg_delivery=("delivery_time_days", "mean"),
        cust_late_ratio=("was_late", "mean"),
        cust_avg_month=("purchase_month", "mean"),
        cust_avg_weekday=("purchase_weekday", "mean"),
        cust_avg_hour=("purchase_hour", "mean"),
        cust_avg_delay=("delay_vs_estimated", "mean"),
        cust_payment_total=("pay_total", "mean"),
        cust_payment_methods=("pay_num_methods", "mean"),
        cust_installments_mean=("pay_installments_mean", "mean"),
    ).reset_index()

    # -------------------------------
    # Features temporales (mensual, trimestral, semestral)
    # -------------------------------
    log("Generando features temporales por ventana...")
    df["year_month"] = df["order_purchase_timestamp"].dt.to_period("M")
    df["quarter"] = df["order_purchase_timestamp"].dt.quarter
    df["semester"] = (df["order_purchase_timestamp"].dt.month - 1)//6 + 1

    # Función auxiliar para crear features de satisfacción en ventana
    def satisfaction_window(df_window, label_prefix):
        # clasificar clientes
        df_window["satisfaction_label"] = np.where(df_window["review_score"] >=4, "satisfecho",
                                          np.where(df_window["review_score"] <=2, "insatisfecho", "neutral"))
        # contar clientes únicos por satisfacción
        summary = df_window.groupby(["customer_id", "satisfaction_label"])["order_id"].count().unstack(fill_value=0)
        summary = summary.rename(columns=lambda x: f"{label_prefix}_{x}")
        return summary

    # Ventanas
    monthly_features = satisfaction_window(df, "month")
    quarterly_features = satisfaction_window(df, "quarter")
    semester_features = satisfaction_window(df, "semester")

    # Merge con customer_features
    customer_features = customer_features.merge(monthly_features, left_on="customer_id", right_index=True, how="left")
    customer_features = customer_features.merge(quarterly_features, left_on="customer_id", right_index=True, how="left")
    customer_features = customer_features.merge(semester_features, left_on="customer_id", right_index=True, how="left")

    # -------------------------------
    # Features polinomiales / ratios
    # -------------------------------
    log("Generando features avanzados (polinomiales y ratios)...")
    for col1 in ["cust_total_spent", "cust_avg_price", "cust_avg_freight"]:
        for col2 in ["cust_avg_delivery", "cust_late_ratio", "cust_installments_mean"]:
            customer_features[f"{col1}_x_{col2}"] = customer_features[col1]*customer_features[col2]

    customer_features["ratio_good_bad_reviews"] = (
        customer_features["cust_num_good_reviews"] / (customer_features["cust_num_bad_reviews"] + 1)
    )

    for col in ["cust_total_spent", "cust_avg_delivery", "cust_review_mean"]:
        customer_features[f"log_{col}"] = np.log1p(customer_features[col])

    log(f"Total features generados: {len(customer_features.columns)}")

    # -------------------------------
    # Selección de features con RandomForest
    # -------------------------------
    log("Entrenando modelo para selección de features...")
    target = customer_features["cust_review_mean"]
    X = customer_features.drop(columns=["cust_review_mean", "customer_id"], errors="ignore")

    # Asegurar numérico
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0)
    target = target.fillna(1)

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X, target)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_features = importances.head(30).index.tolist()

    # Crear dataframe final
    cols_final = ["customer_id"] + [c for c in top_features if c in customer_features.columns] + ["cust_review_mean"]
    df_final = customer_features[cols_final]

    log(f"Shape final de feature table: {df_final.shape}")
    log("=== FIN Feature Engineering Avanzado ===")
    return df_final
