# pipeline.py
# Extracted functions from exploratory_data_alaysis.ipynb for modularity and importing.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

def fetch_and_inspect_data():
    """Fetch UCI dataset and perform initial inspection."""
    online_retail = fetch_ucirepo(id=352)
    df = online_retail.data.original.copy()
    
    print("Columns in df:", df.columns.tolist())
    print(f'Shape: {df.shape}\n')
    print(f"First 5 Rows:\n{df.head()}\n")
    print(f"Data Info:\n")
    df.info()
    print(f"Summary Statistics:\n{df.describe()}")
    print(f"Missing Values:\n{df.isnull().sum()}")
    print(f"\nDate Range (raw): {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    
    return df

def visualize_raw_data(df):
    """Visualize distributions and patterns in raw data."""
    # Quantity distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Quantity'], bins=50)
    plt.title('Distribution of Quantity (Raw Data)')
    plt.xlabel('Quantity')
    plt.ylabel('Count')
    plt.show()

    # UnitPrice distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['UnitPrice'], bins=50)
    plt.title('Distribution of UnitPrice (Raw Data)')
    plt.xlabel('UnitPrice')
    plt.ylabel('Count')
    plt.show()

    # Boxplots
    for col in ['Quantity', 'UnitPrice']:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[col])
        plt.title(f'Box Plot of {col} (Raw Data)')
        plt.show()

    # Top 10 countries
    plt.figure(figsize=(10, 6))
    sns.countplot(y='Country', data=df, order=df['Country'].value_counts().index[:10])
    plt.title('Top 10 Countries by Transaction Count (Raw Data)')
    plt.show()

    # Top 10 products
    print("Top 10 Products by Frequency (Raw Data):")
    print(df['Description'].value_counts().head(10))

    # Outlier min/max
    print(f"\nQuantity Outliers (min/max): {df['Quantity'].min()} / {df['Quantity'].max()}")
    print(f"UnitPrice Outliers (min/max): {df['UnitPrice'].min()} / {df['UnitPrice'].max()}")

    # Correlation heatmap (temp TotalSpend)
    df_temp = df.copy()
    df_temp['TotalSpend'] = df_temp['Quantity'] * df_temp['UnitPrice']
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_temp[['Quantity', 'UnitPrice', 'TotalSpend']].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap (Raw Data)')
    plt.show()

def clean_data(df):
    """Clean the data and verify changes."""
    df_clean = df[df['Quantity'] > 0].copy()
    df_clean = df_clean.dropna(subset=['CustomerID'])
    df_clean = df_clean[df_clean['UnitPrice'] >= 0]
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'], format='%m/%d/%Y %H:%M', errors='coerce')
    df_clean['TotalSpend'] = df_clean['Quantity'] * df_clean['UnitPrice']

    # Verify
    print(f"Shape after cleaning: {df_clean.shape}")
    print(f"Missing Values after cleaning:\n{df_clean.isnull().sum()}")
    print(f"Quantity min/max after cleaning: {df_clean['Quantity'].min()} / {df_clean['Quantity'].max()}")
    print(f"UnitPrice min/max after cleaning: {df_clean['UnitPrice'].min()} / {df_clean['UnitPrice'].max()}")
    print(f"TotalSpend min/max after cleaning: {df_clean['TotalSpend'].min()} / {df_clean['TotalSpend'].max()}")
    print(f"Date Range after cleaning: {df_clean['InvoiceDate'].min()} to {df_clean['InvoiceDate'].max()}")

    return df_clean

def visualize_post_clean(df_clean, rfm):
    """Post-cleaning visualizations with log scales."""
    # Histograms and boxplots
    for col in ['Quantity', 'UnitPrice']:
        plt.figure(figsize=(10, 6))
        sns.histplot(df_clean[col], bins=50, log_scale=True)
        plt.title(f'Distribution of {col} (After Cleaning)')
        plt.xlabel(f'log({col})')
        plt.ylabel('Count')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.show()

    for col in ['Quantity', 'UnitPrice']:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df_clean[col])
        plt.title(f'Box Plot of {col} (After Cleaning)')
        plt.xscale('log')
        plt.show()

    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_clean[['Quantity', 'UnitPrice', 'TotalSpend']].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap (After Cleaning)')
    plt.show()

    # Transactions over time
    df_clean['YearMonth'] = df_clean['InvoiceDate'].dt.to_period('M')
    plt.figure(figsize=(12, 6))
    sns.countplot(x='YearMonth', data=df_clean)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.title('Transactions Over Time (After Cleaning)')
    plt.xticks(rotation=45)
    plt.show()

    # Total Spend by Top 10 Countries
    country_spend = df_clean.groupby('Country')['TotalSpend'].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    country_spend.plot(kind='barh')
    plt.title('Total Spend by Top 10 Countries (After Cleaning)')
    plt.xlabel('Total Spend')
    plt.ylabel('Country')
    plt.show()

    # RFM distributions
    for col in ['Recency', 'Frequency', 'Monetary']:
        plt.figure(figsize=(10, 6))
        sns.histplot(rfm[col], bins=50, log_scale=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(f'log({col})')
        plt.ylabel('Count')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.show()

def compute_rfm(df_clean):
    """Calculate RFM features with CustomerID preserved."""
    today = df_clean['InvoiceDate'].max() + datetime.timedelta(days=1)
    rfm = df_clean.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (today - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalSpend': 'sum'
    }).reset_index()
    rfm = rfm.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalSpend': 'Monetary'})
    return rfm

def split_data(df, test_size=0.15, val_size=0.1765, random_state=42):
    """Split data into train/val/test (approx 70/15/15)."""
    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)
    train, val = train_test_split(train_val, test_size=val_size, random_state=random_state)
    return train, val, test