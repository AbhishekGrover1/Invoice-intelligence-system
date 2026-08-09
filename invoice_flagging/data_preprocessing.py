import sqlite3
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_invoice_data(db_path: str = None) -> pd.DataFrame:
    """
    Load and join vendor invoice data with aggregated purchase data
    from the SQLite database.
    """
    if db_path is None:
        # Resolve relative to this file so it works regardless of cwd
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, "..", "Data", "inventory.db")
        db_path = os.path.abspath(db_path)

    conn = sqlite3.connect(db_path)

    query = """
    WITH purchase_agg AS (
        SELECT 
            p.PONumber,
            COUNT(DISTINCT p.Brand) AS total_brands,
            SUM(p.Quantity) AS total_item_quantity,
            SUM(p.Dollars) AS total_item_dollars,
            AVG(julianday(p.ReceivingDate) - julianday(p.PODate)) AS avg_receiving_delay
        FROM purchases p
        GROUP BY p.PONumber
    )
    SELECT 
        vi.PONumber,
        vi.Quantity AS invoice_quantity,
        vi.Dollars AS invoice_dollars,
        vi.Freight,
        (julianday(vi.InvoiceDate) - julianday(vi.PODate)) AS days_po_to_invoice,
        (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) AS days_to_pay,
        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay
    FROM vendor_invoice vi
    LEFT JOIN purchase_agg pa
      ON vi.PONumber = pa.PONumber
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the target variable for the model.
    Flags an invoice (1) if its billed amount differs from the total
    purchased item cost by more than $5, or if the PO's average
    receiving delay exceeds 10 days.
    """
    # Drop POs that didn't match a purchase record instead of filling
    # with 0, which would misrepresent them as zero-cost purchases
    df = df.dropna()

    price_gap = (df["invoice_dollars"] - df["total_item_dollars"]).abs()
    delayed = df["avg_receiving_delay"] > 10

    df["flag_invoice"] = ((price_gap > 5) | delayed).astype(int)
    return df


def prepare_features(df: pd.DataFrame, features: list, target: str):
    """
    Select features and target variable.
    """
    X = df[features]
    y = df[target]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


def scale_features(X_train, X_test, scaler_path="scaler.pkl"):
    """
    Scales the features using StandardScaler and saves the scaler for future use.
    """
    scaler = StandardScaler()

    # Fit on training data, transform both
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to dataframes to keep column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Save the scaler so it can be loaded during inference/production
    joblib.dump(scaler, scaler_path)

    return X_train_scaled, X_test_scaled