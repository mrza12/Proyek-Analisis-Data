import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency

# Set style for seaborn
sns.set_style(style='dark')

def create_daily_orders_df(df):
    daily_orders_df = df.resample(rule='D', on='order_purchase_timestamp').agg({
        "order_id": "nunique",
        "price": "sum"
    })
    daily_orders_df = daily_orders_df.reset_index()
    daily_orders_df.rename(columns={
        "order_id": "order_count",
        "price": "revenue"
    }, inplace=True)
    
    return daily_orders_df

def create_sum_order_items_df(df):
    sum_order_items_df = df.groupby("product_category_name").order_id.count().sort_values(ascending=False).reset_index()
    sum_order_items_df.rename(columns={"order_id": "frequency"}, inplace=True)
    return sum_order_items_df

def create_bystate_df(df):
    bystate_df = df.groupby(by="customer_state").customer_id.nunique().reset_index()
    bystate_df.rename(columns={"customer_id": "customer_count"}, inplace=True)
    return bystate_df

def create_byreview_df(df):
    byreview_df = df.groupby(["product_category_name", "review_score"]).size().reset_index(name="count")
    return byreview_df

def create_rfm_df(df):
    rfm_df = df.groupby(by="customer_id", as_index=False).agg({
        "order_purchase_timestamp": "max",
        "order_id": "nunique",
        "price": "sum"
    })
    rfm_df.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]
    
    rfm_df["max_order_timestamp"] = rfm_df["max_order_timestamp"].dt.date
    recent_date = df["order_purchase_timestamp"].dt.date.max()
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
    rfm_df.drop("max_order_timestamp", axis=1, inplace=True)
    
    return rfm_df

# Updated function for top 3 product categories trend
def create_top_categories_trend(df):
    top_categories = df.groupby('product_category_name_english')['price'].sum().nlargest(3).index
    trend_df = df[df['product_category_name_english'].isin(top_categories)]
    return trend_df

# New function for customer satisfaction analysis
def create_customer_satisfaction_data(df):
    avg_ratings = df.groupby('product_category_name_english')['review_score'].mean()
    top_3_categories = avg_ratings.sort_values(ascending=False).head(3)
    return top_3_categories, df[['review_score']]

# New function for geographic distribution analysis
def create_geographic_distribution_data(df):
    customer_distribution = df.groupby('customer_state')['customer_id'].nunique().sort_values(ascending=False)
    top_category_per_state = df.groupby(['customer_state', 'product_category_name_english'])['order_id'].count().reset_index()
    top_category_per_state = top_category_per_state.loc[top_category_per_state.groupby('customer_state')['order_id'].idxmax()]
    return customer_distribution, top_category_per_state


# Membaca dataset
all_df = pd.read_csv("all_data.csv")

# Mengubah kolom tanggal ke format datetime
datetime_columns = ["order_purchase_timestamp", "order_delivered_customer_date"]
all_df.sort_values(by="order_purchase_timestamp", inplace=True)
all_df.reset_index(inplace=True)

for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column])

# Membuat filter berdasarkan rentang tanggal
min_date = all_df["order_purchase_timestamp"].min()
max_date = all_df["order_purchase_timestamp"].max()

with st.sidebar:
    st.image("https://assets.digination.id/crop/0x0:0x0/x/photo/2018/10/10/97767959.jpg")
    
    start_date, end_date = st.date_input(
        label='Rentang Waktu', min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_df = all_df[(all_df["order_purchase_timestamp"] >= str(start_date)) & 
                  (all_df["order_purchase_timestamp"] <= str(end_date))]

# Menyiapkan berbagai dataframe
daily_orders_df = create_daily_orders_df(main_df)
sum_order_items_df = create_sum_order_items_df(main_df)
bystate_df = create_bystate_df(main_df)
byreview_df = create_byreview_df(main_df)
rfm_df = create_rfm_df(main_df)
top_categories_trend_df = create_top_categories_trend(main_df)
top_3_categories, reviews_df = create_customer_satisfaction_data(main_df)
customer_distribution, top_category_per_state = create_geographic_distribution_data(main_df)

# Plotting Daily Orders
st.header('E-Commerce Public Dashboard :sparkles:')
st.subheader('Daily Orders')

col1, col2 = st.columns(2)

with col1:
    total_orders = daily_orders_df.order_count.sum()
    st.metric("Total orders", value=total_orders)

with col2:
    total_revenue = format_currency(daily_orders_df.revenue.sum(),"$", locale='es_CO') 
    st.metric("Total Revenue", value=total_revenue)

# Plotting Daily Orders
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    daily_orders_df["order_purchase_timestamp"],
    daily_orders_df["order_count"],
    marker='o', 
    linewidth=2,
    color="#90CAF9"
)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)

st.pyplot(fig)

# Product performance
st.subheader("Best & Worst Performing Product")

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(35, 15))
colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

# Best Performing Product
sns.barplot(x="frequency", y="product_category_name", data=sum_order_items_df.head(5), palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("Number of Sales", fontsize=30)
ax[0].set_title("Best Performing Product", loc="center", fontsize=50)
ax[0].tick_params(axis='y', labelsize=35)
ax[0].tick_params(axis='x', labelsize=30)

# Worst Performing Product
sns.barplot(x="frequency", y="product_category_name", data=sum_order_items_df.sort_values(by="frequency", ascending=True).head(5), palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("Number of Sales", fontsize=30)
ax[1].invert_xaxis()
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_title("Worst Performing Product", loc="center", fontsize=50)
ax[1].tick_params(axis='y', labelsize=35)
ax[1].tick_params(axis='x', labelsize=30)

st.pyplot(fig)

# Create the visualization
st.subheader("Penjualan untuk 3 Kategori Teratas")

total_sales = main_df.groupby('product_category_name_english')['price'].sum().sort_values(ascending=False)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label=total_sales.index[0], value=f"${total_sales.values[0]:,.2f}")
with col2:
    st.metric(label=total_sales.index[1], value=f"${total_sales.values[1]:,.2f}")
with col3:
    st.metric(label=total_sales.index[2], value=f"${total_sales.values[2]:,.2f}")

# New visualization: Customer Satisfaction Analysis
st.subheader("Analisis Kepuasan Pelanggan")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    # Plot 1: Average rating for top 3 categories
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    # colors = sns.color_palette("husl", 1)
    colors = ["#3A6D8C", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
    bars = sns.barplot(x=top_3_categories.index, y=top_3_categories.values, ax=ax1, palette=colors)
    ax1.set_title('Rata-rata Peringkat Ulasan\nuntuk 3 Kategori Produk Teratas', fontsize=16, fontweight='bold')
    # ax1.set_xlabel('Kategori Produk', fontsize=12)
    # ax1.set_ylabel('Rata-rata Peringkat', fontsize=12)
    ax1.set_xticklabels(ax1.get_xticklabels(), ha='center', fontsize=10)
    ax1.tick_params(axis='y', labelsize=10)

    # Add value labels above each bar
    for i, bar in enumerate(bars.patches):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{top_3_categories.values[i]:.2f}',
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    # Plot 2: Distribution of review ratings
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    # review_colors = sns.color_palette("YlOrRd", 1)
    review_colors = ["#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3","#3A6D8C"]
    sns.countplot(x='review_score', data=reviews_df, ax=ax2, palette=review_colors)
    ax2.set_title('Distribusi Peringkat Ulasan', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Peringkat', fontsize=12)
    ax2.set_ylabel('Jumlah Ulasan', fontsize=12)
    ax2.tick_params(axis='both', labelsize=10)

    # Add count labels above each bar
    for i, bar in enumerate(ax2.patches):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{int(bar.get_height())}',
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig2)


# New visualization: Geographic Distribution Analysis
st.subheader("Analisis Distribusi Geografis Pelanggan")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    # Plot distribusi pelanggan per negara bagian
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    colors = ["#3A6D8C", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
    top_3_customers = customer_distribution.head(3)
    sns.barplot(x=top_3_customers.index, y=top_3_customers.values, palette=colors, ax=ax1)
    ax1.set_title('Distribusi Pelanggan per Negara Bagian (Top 3)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Negara Bagian', fontsize=12)
    ax1.set_ylabel('Jumlah Pelanggan', fontsize=12)
    ax1.tick_params(axis='both', labelsize=10)

    # Tambahkan label jumlah di atas setiap bar
    for i, v in enumerate(top_3_customers.values):
        ax1.text(i, v, f'{v:,}', ha='center', va='bottom')

    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    # Plot kategori produk favorit per negara bagian
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='customer_state', y='order_id', hue='product_category_name_english', data=top_category_per_state, ax=ax2)
    ax2.set_title('Kategori Produk Favorit per Negara Bagian', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Negara Bagian', fontsize=12)
    ax2.set_ylabel('Jumlah Penjualan', fontsize=12)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, fontsize=8)
    ax2.tick_params(axis='y', labelsize=10)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig2)

# Visualisasi berdasarkan review
st.subheader("Review Distribution by Product Category")

# Plotting distribusi ulasan
plt.figure(figsize=(10, 5))
fig, ax = plt.subplots(figsize=(12, 6))

# Mengambil 10 kategori teratas berdasarkan jumlah ulasan
top_categories = byreview_df.groupby("product_category_name")["count"].sum().nlargest(10).index

# Filter data untuk 10 kategori teratas
filtered_df = byreview_df[byreview_df["product_category_name"].isin(top_categories)]

sns.heatmap(
    filtered_df.pivot(index="product_category_name", columns="review_score", values="count").fillna(0),
    annot=True,
    fmt="g",
    cmap="YlOrRd",
    ax=ax
)

ax.set_title("Review Distribution for Top 10 Product Categories", fontsize=20)
ax.set_xlabel("Review Score", fontsize=15)
ax.set_ylabel("Product Category", fontsize=15)
ax.tick_params(axis='x', labelsize=12, rotation=0)
ax.tick_params(axis='y', labelsize=12)

st.pyplot(fig)

# Menambahkan penjelasan
st.write("""
This heatmap shows the distribution of review scores for the top 10 product categories based on the total number of reviews.
The intensity of the color represents the frequency of reviews, with darker colors indicating a higher number of reviews.
""")
