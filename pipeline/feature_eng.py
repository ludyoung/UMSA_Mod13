# pipeline/feature_eng.py
import pandas as pd
from pipeline.utils import log

def generar_features(dfs: dict) -> pd.DataFrame:
    """
    Genera features de cliente para análisis de satisfacción.
    Retorna DataFrame con features agregados por customer_id.
    """

    log("=== INICIO Feature Engineering ===")

    # Cargar tablas necesarias
    log("Cargando tablas base para features...")
    orders = dfs["orders"]
    order_items = dfs["order_items"]
    reviews = dfs["reviews"]
    products = dfs["products"]
    sellers = dfs["sellers"]
    payments = dfs["payments"]
    customers = dfs["customers"]
    geo = dfs["geolocation"]
    cat_trans = dfs["category_translation"]
    log("Tablas cargadas: orders, order_items, reviews, products, sellers, payments, customers, geolocation, category_translation")

    # Merge maestro
    log("Realizando merge de tablas para construir dataset base...")
    df = orders.merge(order_items, on="order_id", how="left") \
               .merge(reviews, on="order_id", how="left") \
               .merge(products, on="product_id", how="left") \
               .merge(sellers, on="seller_id", how="left") \
               .merge(payments, on="order_id", how="left") \
               .merge(customers, on="customer_id", how="left")
    log(f"Merged dataset shape: {df.shape}")

    # ------------------------
    # Features básicos
    # ------------------------
    log("Creando features de tiempo y retraso de entrega...")
    df = df.dropna(subset=["order_purchase_timestamp", "order_delivered_customer_date"])
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"], errors="coerce")

    df["delivery_time_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
    df["estimated_delay_days"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days
    df["was_late"] = (df["estimated_delay_days"] > 0).astype(int)
    df["order_month"] = df["order_purchase_timestamp"].dt.month
    df["order_quarter"] = df["order_purchase_timestamp"].dt.quarter
    df["order_semester"] = df["order_purchase_timestamp"].dt.month.apply(lambda x: 1 if x <=6 else 2)
    log("Features de tiempo generados: delivery_time_days, estimated_delay_days, was_late, order_month, order_quarter, order_semester")

    # Features por cliente
    log("Agregando features por customer_id...")
    customer_features = df.groupby("customer_id").agg(
        cust_total_orders=("order_id", "count"),
        cust_total_spent=("price", "sum"),
        cust_avg_price=("price", "mean"),
        cust_avg_freight=("freight_value", "mean"),
        cust_num_products=("product_id", "nunique"),
        cust_review_mean=("review_score", "mean"),
        cust_review_std=("review_score", "std"),
        cust_num_bad_reviews=("review_score", lambda x: (x <=2).sum()),
        cust_num_good_reviews=("review_score", lambda x: (x >=4).sum()),
        cust_avg_delivery=("delivery_time_days", "mean"),
        cust_late_ratio=("was_late", "mean"),
        cust_avg_order_month=("order_month", "mean"),
        cust_avg_order_quarter=("order_quarter", "mean"),
        cust_avg_order_semester=("order_semester", "mean")
    ).reset_index()

    log(f"Features generados por cliente: {list(customer_features.columns)}")
    log("=== FIN Feature Engineering ===")

    return customer_features
