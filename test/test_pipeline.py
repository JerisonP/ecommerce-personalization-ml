# test/test_pipeline.py
import sys
sys.path.append('..')  # Add root to path for imports

import pandas as pd
import numpy as np
from my_library.pipeline import fetch_and_inspect_data, clean_data, compute_rfm, split_data  # Package.module import

# Updated mock data for tests (4 rows, cleans to 3 rows: remove row 2 nan; all positive Q/U)
mock_df = pd.DataFrame({
    'Quantity': [1, 2, 3, 4],
    'UnitPrice': [1.0, 2.0, 3.0, 4.0],
    'CustomerID': [1, 2, np.nan, 3],
    'InvoiceDate': ['12/1/2010 8:26', '12/1/2010 8:26', '12/1/2010 8:26', '12/1/2010 8:26'],
    'InvoiceNo': ['A', 'B', 'C', 'D'],
    'TotalSpend': [1, 4, 9, 16]  # Pre-calculated for test
})

def test_fetch_and_inspect_data():
    df = fetch_and_inspect_data()
    assert df.shape[0] > 500000, "Fetched data should have ~541k rows"
    assert 'InvoiceNo' in df.columns, "InvoiceNo missing"

def test_clean_data():
    df_clean = clean_data(mock_df)
    assert df_clean.shape[0] == 3, "Cleaning should remove nulls (3 rows left)"
    assert df_clean['Quantity'].min() > 0, "No negative Quantity"
    assert 'TotalSpend' in df_clean.columns, "TotalSpend added"

def test_compute_rfm():
    mock_df_clean = clean_data(mock_df)
    mock_df_clean['InvoiceDate'] = pd.to_datetime(mock_df_clean['InvoiceDate'])
    rfm = compute_rfm(mock_df_clean)
    assert rfm.shape[1] == 4, "RFM has CustomerID + 3 features"
    assert rfm['Recency'].min() >= 0, "Recency non-negative"

def test_split_data():
    mock_df_clean = clean_data(mock_df)
    # Use smaller test_size for small mock (to avoid n_train=0)
    train, val, test = split_data(mock_df_clean, test_size=0.3, val_size=0.3)
    assert len(train) + len(val) + len(test) == len(mock_df_clean), "Splits sum to original"
    assert len(test) > 0, "Test set not empty"