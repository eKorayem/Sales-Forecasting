import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Page setup
st.set_page_config(page_title="Sales Forecast Dashboard", page_icon="📊", layout="wide")

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        color: #0D47A1;
        margin-top: 15px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Smart Sales Forecast Dashboard</div>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("Sales.csv")
    df.columns = df.columns.str.strip().str.replace(r"\s+", "_", regex=True)
    
    # Standardize column names that might differ in casing or spacing
    column_mapping = {
        'Customer_ID': 'Customer ID',
        'Product_ID': 'Product ID',
        'Order_ID': 'Order ID'
    }
    
    # Apply mapping if columns exist in different format
    for new_col, old_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)
        elif new_col.replace('_', ' ') in df.columns:
            df.rename(columns={new_col.replace('_', ' '): new_col}, inplace=True)
    
    # Remove "popularity" and "Order count" columns if they exist
    columns_to_remove = ['popularity', 'Popularity', 'order_count', 'Order_count', 'Order_Count', 'Order count']
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    df['Ship_Date'] = pd.to_datetime(df['Ship_Date'])
    df['Delivery_Time'] = (df['Ship_Date'] - df['Order_Date']).dt.days
    df['Discount'] = df['Discount'] * 100
    
    df.fillna(df.mode().iloc[0], inplace=True)
    return df

@st.cache_resource
def train_model(df):
    # Prepare data for training
    # Select features and target
    features = df[['Delivery_Time', 'Quantity', 'Category', 'Sub-Category', 'Discount', 'Profit']].copy()
    
    # Encode categorical features
    le_category = LabelEncoder()
    le_subcategory = LabelEncoder()
    
    features['Category_Encoded'] = le_category.fit_transform(features['Category'])
    features['SubCategory_Encoded'] = le_subcategory.fit_transform(features['Sub-Category'])
    
    # Store mapping for reference
    category_mapping = dict(zip(features['Category'], features['Category_Encoded']))
    subcategory_mapping = dict(zip(features['Sub-Category'], features['SubCategory_Encoded']))
    
    # Select numerical features
    X = features[['Delivery_Time', 'Quantity', 'Category_Encoded', 'SubCategory_Encoded', 
                 'Discount', 'Profit']]
    
    # Target variable (log transform to handle skewness)
    y = np.log1p(df['Sales'])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # RF model with the provided parameters
    params = {'max_depth': 11, 'n_estimators' : 150}

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    model.fit(X_train_scaled, y_train)
    
    return model, scaler, le_category, le_subcategory

try:
    # Load data
    df = load_data()
    st.sidebar.success("Data loaded successfully!")
    
    # Train model (cached)
    with st.spinner('Training model (this will be cached after first run)...'):
        model, scaler, le_category, le_subcategory = train_model(df)
    
    # Sidebar Filters
    st.sidebar.markdown("## Filters")
    min_date, max_date = df['Order_Date'].min().date(), df['Order_Date'].max().date()
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
    if len(date_range) == 2:
        df = df[(df['Order_Date'].dt.date >= date_range[0]) & (df['Order_Date'].dt.date <= date_range[1])]

    category_options = sorted(df['Category'].unique())
    subcat_options = sorted(df['Sub-Category'].unique())

    selected_category = st.sidebar.selectbox("Category", ['All'] + category_options)
    if selected_category != 'All':
        df = df[df['Category'] == selected_category]
        # Update subcategory options based on selected category
        subcat_options = sorted(df['Sub-Category'].unique())

    selected_subcategory = st.sidebar.selectbox("Sub-Category", ['All'] + subcat_options)
    if selected_subcategory != 'All':
        df = df[df['Sub-Category'] == selected_subcategory]

    selected_state = st.sidebar.selectbox("State", ['All'] + sorted(df['State'].unique()))
    if selected_state != 'All':
        df = df[df['State'] == selected_state]

    selected_ship_mode = st.sidebar.selectbox("Shipping Mode", ['All'] + sorted(df['Ship_Mode'].unique()))
    if selected_ship_mode != 'All':
        df = df[df['Ship_Mode'] == selected_ship_mode]

    # Metrics
    st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Sales", f"${df['Sales'].sum():,.2f}")
    k2.metric("Total Profit", f"${df['Profit'].sum():,.2f}")
    k3.metric("Profit Margin", f"{df['Profit'].sum() / df['Sales'].sum() * 100:.2f}%")
    k4.metric("Avg. Delivery Time", f"{df['Delivery_Time'].mean():.1f} days")

    # Interactive Sales Predictor
    st.markdown('<div class="section-header">Interactive Sales Predictor</div>', unsafe_allow_html=True)

    # Display available category and subcategory classes from the model
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Information")
    with st.sidebar.expander("Available Categories"):
        st.write(", ".join(sorted(le_category.classes_)))
    with st.sidebar.expander("Available Sub-Categories"):
        st.write(", ".join(sorted(le_subcategory.classes_)))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        delivery = st.slider("Delivery Time", 1, 10, 3)
    with col2:
        quantity = st.slider("Quantity", 1, 50, 5)
    with col3:
        # Use the categories that were trained in the model
        category_label = st.selectbox("Category", sorted(le_category.classes_))
    with col4:
        # Filter subcategories by the selected category if possible
        # Get all subcategories first
        all_subcats = sorted(le_subcategory.classes_)
        
        # If the data has a relationship between category and subcategory, filter by it
        related_subcats = df[df['Category'] == category_label]['Sub-Category'].unique()
        
        # Use related subcategories if available, otherwise use all
        if len(related_subcats) > 0:
            subcategory_options = sorted(related_subcats)
        else:
            subcategory_options = all_subcats
            
        subcategory_label = st.selectbox("Sub-Category", subcategory_options)

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        discount = st.slider("Discount (%)", 0.0, 100.0, 10.0)
    with col6:
        profit = st.number_input("Profit ($)", value=50.0)

    if st.button("Predict Sales"):
        try:
            # Get the encoded values for the selected category and subcategory
            if category_label in le_category.classes_ and subcategory_label in le_subcategory.classes_:
                category_encoded = le_category.transform([category_label])[0]
                subcategory_encoded = le_subcategory.transform([subcategory_label])[0]
                
                input_values = np.array([[delivery, quantity, category_encoded, subcategory_encoded, discount, profit]])
                input_scaled = scaler.transform(input_values)
                log_prediction = model.predict(input_scaled)[0]
                predicted_sales = np.expm1(log_prediction)
                st.success(f"Predicted Sales: ${predicted_sales:,.2f}")
                
            else:
                if category_label not in le_category.classes_:
                    st.error(f"Category '{category_label}' was not present in the training data.")
                if subcategory_label not in le_subcategory.classes_:
                    st.error(f"Sub-Category '{subcategory_label}' was not present in the training data.")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Add visualizations
    st.markdown('<div class="section-header">Sales Analysis</div>', unsafe_allow_html=True)
    
    # Time series analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales Over Time")
        sales_by_date = df.groupby(df['Order_Date'].dt.to_period('M')).agg({'Sales': 'sum'}).reset_index()
        sales_by_date['Order_Date'] = sales_by_date['Order_Date'].dt.to_timestamp()
        
        fig = px.line(sales_by_date, x='Order_Date', y='Sales', 
                    labels={'Order_Date': 'Date', 'Sales': 'Total Sales ($)'},
                    title='Monthly Sales Trend')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Category Performance")
        cat_sales = df.groupby('Category').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
        fig = px.bar(cat_sales, x='Category', y=['Sales', 'Profit'], 
                    title='Sales and Profit by Category',
                    barmode='group',
                    color_discrete_sequence=['darkblue', 'orange'])
        fig.update_layout(height=400, xaxis_tickangle=35, 
                        legend_title_text='Metric',
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)
    
    # Add profit/loss by state analysis (similar to your matplotlib example)
    st.subheader("Profit Analysis by State")
    
    # Calculate profit metrics by state
    state_profit = df.groupby('State')['Profit'].sum().reset_index()
    
    # Separate profitable and non-profitable states
    profitable_states = state_profit[state_profit['Profit'] > 0].sort_values('Profit', ascending=False)
    profitable_states['share_of_profit'] = profitable_states['Profit'] / profitable_states['Profit'].sum()
    profitable_states = profitable_states.head(10)  # Top 10 profitable states
    
    no_profit_states = state_profit[state_profit['Profit'] <= 0].sort_values('Profit')
    no_profit_states['share_of_loss'] = abs(no_profit_states['Profit']) / abs(no_profit_states['Profit'].sum())
    
    # Create two horizontal bar charts side by side
    fig = sp.make_subplots(rows=1, cols=2, 
                          subplot_titles=("Share of Total Loss by State (%)", 
                                         "Share of Total Profit by State (%) - Top 10"),
                          horizontal_spacing=0.15)
    
    # Loss chart (left)
    fig.add_trace(
        go.Bar(
            y=no_profit_states['State'],
            x=no_profit_states['share_of_loss'],
            orientation='h',
            marker=dict(color=no_profit_states['share_of_loss'], 
                      colorscale='OrRd_r'),
            text=[f"{x*100:.1f}%" for x in no_profit_states['share_of_loss']],
            textposition='outside',
            hoverinfo='text',
            hovertext=[f"{state}: {loss*100:.1f}%" for state, loss in 
                    zip(no_profit_states['State'], no_profit_states['share_of_loss'])],
        ),
        row=1, col=1
    )
    
    # Profit chart (right)
    fig.add_trace(
        go.Bar(
            y=profitable_states['State'],
            x=profitable_states['share_of_profit'],
            orientation='h',
            marker=dict(color=profitable_states['share_of_profit'], 
                      colorscale='Blues_r'),
            text=[f"{x*100:.1f}%" for x in profitable_states['share_of_profit']],
            textposition='outside',
            hoverinfo='text',
            hovertext=[f"{state}: {profit*100:.1f}%" for state, profit in 
                    zip(profitable_states['State'], profitable_states['share_of_profit'])],
        ),
        row=1, col=2
    )
    
    # Update layout to match the clean style from your example
    fig.update_layout(
        height=500,
        showlegend=False,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False),
        xaxis2=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False),
        yaxis=dict(showgrid=False),
        yaxis2=dict(showgrid=False),
        plot_bgcolor='white',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sub-Category Sales and Profit (bar chart)
    st.subheader("Sales and Profit by Sub-Category")
    subcat_sales = df.groupby('Sub-Category')[['Sales', 'Profit']].sum().reset_index().sort_values('Sales', ascending=False)
    
    fig = px.bar(subcat_sales, x='Sub-Category', y=['Sales', 'Profit'],
                barmode='group',
                color_discrete_sequence=['darkblue', 'orange'],
                title='Total Sales and Profit by Sub-Category')
    
    fig.update_layout(
        height=500,
        xaxis_tickangle=35,
        legend_title_text='Metric',
        xaxis=dict(showgrid=False, title=None),
        yaxis=dict(showgrid=False, title='Amount ($)'),
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Delivery Mode Analysis with pie chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Orders by Ship Mode")
        ship_mode_counts = df['Ship_Mode'].value_counts()
        
        fig = px.pie(
            values=ship_mode_counts.values,
            names=ship_mode_counts.index,
            title='Distribution of Orders by Ship Mode',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        fig.update_traces(
            textposition='outside',
            textinfo='percent+label',
            pull=[0.03] * len(ship_mode_counts)
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Quantity Ordered by Month")
        df['Month'] = df['Order_Date'].dt.strftime('%b')
        df['Month_num'] = df['Order_Date'].dt.month
        month_quantity = df.groupby(['Month', 'Month_num'])['Quantity'].sum().reset_index()
        month_quantity = month_quantity.sort_values('Month_num')
        
        fig = px.bar(
            month_quantity, 
            x='Month', 
            y='Quantity', 
            title='Total Quantity Ordered by Month',
            color_discrete_sequence=['skyblue']
        )
        
        fig.update_layout(
            height=400,
            xaxis=dict(title=None, showgrid=False),
            yaxis=dict(title='Quantity', showgrid=False),
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

except FileNotFoundError:
    st.error("Error: Required file not found. Ensure 'Sales.csv' is present.")
except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")