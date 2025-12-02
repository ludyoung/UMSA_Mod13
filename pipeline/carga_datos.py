# pipeline/carga_datos.py
import pandas as pd
import os
from pipeline.utils import log

def ejecutar_carga():
    try:
        base_path = "./dataset"
        dfs = {
            "customers": pd.read_csv(os.path.join(base_path, "olist_customers_dataset.csv")),
            "geolocation": pd.read_csv(os.path.join(base_path, "olist_geolocation_dataset.csv")),
            "order_items": pd.read_csv(os.path.join(base_path, "olist_order_items_dataset.csv")),
            "payments": pd.read_csv(os.path.join(base_path, "olist_order_payments_dataset.csv")),
            "reviews": pd.read_csv(os.path.join(base_path, "olist_order_reviews_dataset.csv")),
            "orders": pd.read_csv(os.path.join(base_path, "olist_orders_dataset.csv")),
            "products": pd.read_csv(os.path.join(base_path, "olist_products_dataset.csv")),
            "sellers": pd.read_csv(os.path.join(base_path, "olist_sellers_dataset.csv")),
            "category_translation": pd.read_csv(os.path.join(base_path, "product_category_name_translation.csv"))
        }
        rows_map = {k: len(v) for k, v in dfs.items()}
        log(f"============== Tablas cargadas exitosamente: {rows_map}")
        return dfs, rows_map

    except Exception as e:
        log(f"!! ============== Error al cargar datos: {e}")
        return None, None
