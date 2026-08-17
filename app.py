import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Coffee Shop Analytics", layout="wide", page_icon="☕")
st.title("☕ Coffee Shop Sales Dashboard")
st.markdown("Interact with the sidebar to filter sales data and discover business insights.")

# 2. Data Loading & Mock Data Generation
@st.cache_data
def load_data():
    try:
        # Attempt to load the coffee dataset
        df = pd.read_csv("coffee_sales_data.csv")
    except FileNotFoundError:
        # Creating a realistic mock coffee dataset if file is missing
        np.random.seed(42)
        data = {
            'TransactionID': np.random.randint(100001, 105000, 2000),
            'OrderDate': pd.date_range(start='2025-01-01', periods=2000, freq='30min'),
            'CustomerName': np.random.choice(['Alex Smith', 'Maria Garcia', 'James Wilson', 'Emma Brown', 'John Doe'], 2000),
            'CoffeeType': np.random.choice(['Espresso', 'Latte', 'Cappuccino', 'Americano', 'Macchiato', 'Mocha'], 2000),
            'CoffeeVariety': np.random.choice(['Arabica', 'Robusta', 'Blend'], 2000),
            'Size': np.random.choice(['Small', 'Medium', 'Large'], 2000, p=[0.3, 0.5, 0.2]),
            'QuantityOrdered': np.random.randint(1, 5, 2000),
            'UnitPrice': np.random.choice([2.50, 3.75, 4.20, 3.00, 4.00, 4.50], 2000),
            'CustomerType': np.random.choice(['Member', 'Walk-in'], 2000, p=[0.4, 0.6]),
            'PaymentMethod': np.random.choice(['Cash', 'Credit Card', 'Mobile Wallet'], 2000)
        }
        df = pd.DataFrame(data)
    
    # Data Processing & Cleaning
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df['TotalBillAmount'] = df['QuantityOrdered'] * df['UnitPrice']
    df['Month'] = df['OrderDate'].dt.strftime('%Y-%m')
    return df

df = load_data()

# 3. Sidebar Filters
st.sidebar.header("☕ Filter Options")

# Coffee Type Filter
coffee_types = st.sidebar.multiselect(
    "Select Coffee Types", 
    options=df['CoffeeType'].unique(), 
    default=df['CoffeeType'].unique()
)

# Customer Type Filter
customer_types = st.sidebar.multiselect(
    "Select Customer Type", 
    options=df['CustomerType'].unique(), 
    default=df['CustomerType'].unique()
)

# Apply filters to dataframe
filtered_df = df[
    (df['CoffeeType'].isin(coffee_types)) & 
    (df['CustomerType'].isin(customer_types))
]

# 4. Top-Level Key Metrics
total_revenue = filtered_df['TotalBillAmount'].sum()
total_orders = filtered_df['TransactionID'].nunique()
avg_bill_value = total_revenue / total_orders if total_orders > 0 else 0

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="💰 Total Sales Revenue", value=f"${total_revenue:,.2f}")
with col2:
    st.metric(label="📋 Total Orders Cups", value=f"{total_orders:,}")
with col3:
    st.metric(label="☕ Avg Bill Value", value=f"${avg_bill_value:,.2f}")

st.markdown("---")

# 5. Visual Charts (Two-Column Layout)
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("📈 Monthly Revenue Trend")
    monthly_sales = filtered_df.groupby('Month')['TotalBillAmount'].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(data=monthly_sales, x='Month', y='TotalBillAmount', marker='o', color='#8B4513', ax=ax)
    plt.xticks(rotation=45)
    plt.ylabel("Revenue ($)")
    plt.xlabel("Timeline")
    st.pyplot(fig)

with chart_col2:
    st.subheader("🏆 Most Popular Coffee Types")
    top_coffee = filtered_df.groupby('CoffeeType')['QuantityOrdered'].sum().sort_values(ascending=False).reset_index()
    
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=top_coffee, x='QuantityOrdered', y='CoffeeType', palette='YlOrBr_r', ax=ax)
    plt.xlabel("Total Cups Sold")
    plt.ylabel("Coffee Type")
    st.pyplot(fig)

# 6. Data Table Preview
st.subheader("📋 Coffee Sales Raw Data Explorer")
st.dataframe(filtered_df.head(100), use_container_width=True)