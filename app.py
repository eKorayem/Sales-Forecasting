import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set page configuration
st.set_page_config(
    page_title="Superstore Sales Dashboard",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS for styling
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
    .small-text {
        font-size: 14px;
        color: #616161;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">Superstore Sales Dashboard</div>', unsafe_allow_html=True)

# Function to load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('Sales.csv')
    df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
    
    # Handle missing values
    df.fillna(df.mode().iloc[0], inplace=True)
    
    # Process columns
    df['Discount'] = (df['Discount']) * 100
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    df['Delivery Time'] = (df['Ship Date'] - df['Order Date']).dt.days
    
    # Remove outliers
    numeric_df = df.select_dtypes(include=['int', 'float'])
    outlier_indices = set()

    for column in numeric_df.columns:
        Q1 = numeric_df[column].quantile(0.25)
        Q3 = numeric_df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 5 * IQR
        upper_bound = Q3 + 5 * IQR

        outliers_mask = (numeric_df[column] < lower_bound) | (numeric_df[column] > upper_bound)
        outliers = numeric_df[column][outliers_mask]
        
        if not outliers.empty:
            outlier_indices.update(outliers.index)

    # Drop all rows with any outlier
    df.drop(index=outlier_indices, inplace=True)
    
    return df

# Load data
try:
    df = load_data()
    
    # Display data loading status
    st.sidebar.success("Data loaded successfully!")
    
    # Sidebar filters
    st.sidebar.markdown("## Filters")
    
    # Date range filter
    min_date = df['Order Date'].min().date()
    max_date = df['Order Date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[(df['Order Date'].dt.date >= start_date) & 
                         (df['Order Date'].dt.date <= end_date)]
    else:
        filtered_df = df
    
    # Category filter
    category_list = ['All'] + list(df['Category'].unique())
    selected_category = st.sidebar.selectbox("Select Category", category_list)
    
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]
    
    # State filter
    state_list = ['All'] + list(df['State'].unique())
    selected_state = st.sidebar.selectbox("Select State", state_list)
    
    if selected_state != 'All':
        filtered_df = filtered_df[filtered_df['State'] == selected_state]
    
    # Shipping mode filter
    ship_mode_list = ['All'] + list(df['Ship Mode'].unique())
    selected_ship_mode = st.sidebar.selectbox("Select Shipping Mode", ship_mode_list)
    
    if selected_ship_mode != 'All':
        filtered_df = filtered_df[filtered_df['Ship Mode'] == selected_ship_mode]
    
    # Dashboard Main Content
    # KPI Cards
    st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Sales",
            value=f"${filtered_df['Sales'].sum():,.2f}",
            delta=f"{filtered_df['Sales'].sum() / df['Sales'].sum() * 100:.1f}%" if selected_category != 'All' or selected_state != 'All' or selected_ship_mode != 'All' else None
        )
    
    with col2:
        st.metric(
            label="Total Profit",
            value=f"${filtered_df['Profit'].sum():,.2f}",
            delta=f"{filtered_df['Profit'].sum() / df['Profit'].sum() * 100:.1f}%" if selected_category != 'All' or selected_state != 'All' or selected_ship_mode != 'All' else None
        )
    
    with col3:
        profit_margin = filtered_df['Profit'].sum() / filtered_df['Sales'].sum() * 100
        st.metric(
            label="Profit Margin",
            value=f"{profit_margin:.2f}%"
        )
    
    with col4:
        st.metric(
            label="Average Delivery Time",
            value=f"{filtered_df['Delivery Time'].mean():.1f} days"
        )
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Sales Analysis", 
        "Geographical Analysis", 
        "Profit Analysis", 
        "Time Series",
        "Predictive Analytics"
    ])
    
    # Tab 1: Sales Analysis
    with tab1:
        st.markdown('<div class="section-header">Sales Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category sales and profit pie chart
            df1 = filtered_df[['Category', 'Sales', 'Profit']]
            df1 = df1.groupby(by='Category', as_index=False).sum()
            
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=df1['Category'],
                values=df1['Sales'],
                hole=0.4,
                name="Sales"
            ))
            
            fig.update_layout(
                title="Sales Distribution by Category",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Profit distribution by category
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=df1['Category'],
                values=df1['Profit'],
                hole=0.4,
                name="Profit"
            ))
            
            fig.update_layout(
                title="Profit Distribution by Category",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Sub-category sales and profit
        st.markdown('<div class="section-header">Sub-Category Performance</div>', unsafe_allow_html=True)
        
        df_sub = filtered_df.groupby('Sub-Category')[['Sales', 'Profit']].sum().sort_values('Sales', ascending=False).reset_index()
        df_sub['Profit Margin'] = df_sub['Profit'] / df_sub['Sales'] * 100
        
        fig = px.bar(
            df_sub,
            x='Sub-Category',
            y=['Sales', 'Profit'],
            barmode='group',
            title='Sales and Profit by Sub-Category',
            labels={'value': 'Amount ($)', 'variable': 'Metric'},
            color_discrete_map={'Sales': 'darkblue', 'Profit': 'orange'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Profit margin by sub-category
        fig = px.bar(
            df_sub.sort_values('Profit Margin'),
            x='Profit Margin',
            y='Sub-Category',
            orientation='h',
            title='Profit Margin by Sub-Category (%)',
            color='Profit Margin',
            color_continuous_scale='RdYlGn',
            labels={'Profit Margin': 'Profit Margin (%)'}
        )
        
        fig.add_vline(
            x=filtered_df['Profit'].sum() / filtered_df['Sales'].sum() * 100,
            line_dash="dash",
            line_color="red",
            annotation_text="Average"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Geographical Analysis
    with tab2:
        st.markdown('<div class="section-header">Geographical Analysis</div>', unsafe_allow_html=True)
        
        # State-wise performance
        df_states = filtered_df.groupby('State')[['Sales', 'Profit']].sum().reset_index()
        df_states['Profit Margin'] = df_states['Profit'] / df_states['Sales'] * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top states by sales
            top_sales_states = df_states.sort_values('Sales', ascending=False).head(10)
            fig = px.bar(
                top_sales_states,
                x='Sales',
                y='State',
                orientation='h',
                title='Top 10 States by Sales',
                color='Sales',
                color_continuous_scale='Blues',
                labels={'Sales': 'Sales ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # States by profit margin
            df_states_sorted = df_states.sort_values('Profit', ascending=False)
            profitable_states = df_states_sorted[df_states_sorted['Profit'] > 0].head(10)
            loss_states = df_states_sorted[df_states_sorted['Profit'] < 0].sort_values('Profit').head(10)
            
            fig = px.bar(
                profitable_states,
                x='Profit',
                y='State',
                orientation='h',
                title='Top 10 Most Profitable States',
                color='Profit',
                color_continuous_scale='Greens',
                labels={'Profit': 'Profit ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if not loss_states.empty:
            fig = px.bar(
                loss_states,
                x='Profit',
                y='State',
                orientation='h',
                title='Top 10 States with Losses',
                color='Profit',
                color_continuous_scale='Reds_r',
                labels={'Profit': 'Loss ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Profit Analysis
    with tab3:
        st.markdown('<div class="section-header">Profit Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Discount distribution
            fig = px.histogram(
                filtered_df,
                x='Discount',
                nbins=20,
                title='Discount Distribution',
                color_discrete_sequence=['darkcyan']
            )
            fig.update_layout(
                xaxis_title='Discount (%)',
                yaxis_title='Count'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Discount vs. Profit
            fig = px.scatter(
                filtered_df,
                x='Discount',
                y='Profit',
                color='Category',
                title='Discount Impact on Profit',
                labels={'Discount': 'Discount (%)', 'Profit': 'Profit ($)'}
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        # Shipping analysis
        st.markdown('<div class="section-header">Shipping Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average Sales & Profit by Shipping Mode
            shipping_perf = filtered_df.groupby('Ship Mode')[['Sales', 'Profit']].mean().reset_index()
            fig = px.bar(
                shipping_perf,
                x='Ship Mode',
                y=['Sales', 'Profit'],
                barmode='group',
                title='Average Sales & Profit by Shipping Mode',
                color_discrete_map={'Sales': '#f39c12', 'Profit': '#1abc9c'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Order count by shipping mode
            shipping_counts = filtered_df['Ship Mode'].value_counts().reset_index()
            shipping_counts.columns = ['Ship Mode', 'Count']
            
            fig = px.bar(
                shipping_counts,
                x='Ship Mode',
                y='Count',
                title='Order Count by Shipping Mode',
                color='Ship Mode'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Delivery time analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Delivery time distribution
            fig = px.histogram(
                filtered_df,
                x='Delivery Time',
                nbins=12,
                title='Delivery Time Distribution',
                color_discrete_sequence=['darkcyan']
            )
            fig.update_layout(
                xaxis_title='Delivery Time (Days)',
                yaxis_title='Count'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Delivery time pie chart
            delivery_counts = filtered_df['Delivery Time'].value_counts().reset_index()
            delivery_counts.columns = ['Delivery Time', 'Count']
            
            fig = px.pie(
                delivery_counts,
                values='Count',
                names='Delivery Time',
                title='Delivery Time Distribution (Days)'
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Time Series Analysis
    with tab4:
        st.markdown('<div class="section-header">Time Series Analysis</div>', unsafe_allow_html=True)
        
        # Monthly sales and profit
        monthly_sales = filtered_df.set_index('Order Date').resample('M')[['Sales', 'Profit']].sum().reset_index()
        monthly_sales['Order Date'] = monthly_sales['Order Date'].dt.strftime('%Y-%m')
        
        fig = px.line(
            monthly_sales,
            x='Order Date',
            y=['Sales', 'Profit'],
            title='Monthly Sales and Profit Over Time',
            markers=True,
            labels={'value': 'Amount ($)', 'variable': 'Metric'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top customers
        st.markdown('<div class="section-header">Top Customers</div>', unsafe_allow_html=True)
        top_customers = filtered_df.groupby('Customer Name')['Sales'].sum().sort_values(ascending=False).head(10).reset_index()
        
        fig = px.bar(
            top_customers,
            x='Sales',
            y='Customer Name',
            orientation='h',
            title='Top 10 Customers by Total Sales',
            color='Sales',
            color_continuous_scale='viridis',
            labels={'Sales': 'Sales ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Predictive Analytics
    with tab5:
        st.markdown('<div class="section-header">Profit Prediction Model</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature correlation heatmap
            corr_data = filtered_df.select_dtypes(include=['int', 'float']).corr()
            fig = px.imshow(
                corr_data,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title='Feature Correlation Heatmap'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Model training explanation
            st.markdown("""
            ### Profit Prediction
            
            The model predicts profit based on:
            - Sales amount
            - Discount percentage
            - Delivery time
            
            Try the interactive predictor below to estimate profit for different scenarios.
            """)
            
            # Train model once for prediction
            X = filtered_df[['Sales', 'Discount', 'Delivery Time']]
            y = filtered_df['Profit']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Linear Regression
            lr_model = LinearRegression()
            lr_model.fit(X_train_scaled, y_train)
            lr_pred = lr_model.predict(X_test_scaled)
            lr_r2 = r2_score(y_test, lr_pred)
            lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
            
            # Random Forest
            rf_model = RandomForestRegressor(
                max_depth=20,
                bootstrap=False,
                min_samples_leaf=1,
                max_features='sqrt',
                min_samples_split=2,
                n_estimators=100,
                random_state=42
            )
            rf_model.fit(X_train_scaled, y_train)
            rf_pred = rf_model.predict(X_test_scaled)
            rf_r2 = r2_score(y_test, rf_pred)
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
            
            # Display model metrics
            st.markdown("#### Model Performance")
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                st.metric("Linear Regression R²", f"{lr_r2:.4f}")
                st.metric("Linear Regression RMSE", f"${lr_rmse:.2f}")
            
            with metric_col2:
                st.metric("Random Forest R²", f"{rf_r2:.4f}")
                st.metric("Random Forest RMSE", f"${rf_rmse:.2f}")
        
        # Interactive profit predictor
        st.markdown('<div class="section-header">Interactive Profit Predictor</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sales_input = st.number_input(
                "Sales Amount ($)",
                min_value=float(filtered_df['Sales'].min()),
                max_value=float(filtered_df['Sales'].max()),
                value=float(filtered_df['Sales'].median())
            )
        
        with col2:
            discount_input = st.slider(
                "Discount (%)",
                min_value=float(filtered_df['Discount'].min()),
                max_value=float(filtered_df['Discount'].max()),
                value=float(filtered_df['Discount'].median())
            )
        
        with col3:
            delivery_input = st.slider(
                "Delivery Time (Days)",
                min_value=int(filtered_df['Delivery Time'].min()),
                max_value=int(filtered_df['Delivery Time'].max()),
                value=int(filtered_df['Delivery Time'].median())
            )
        
        # Make prediction
        if st.button("Predict Profit"):
            input_data = np.array([[sales_input, discount_input, delivery_input]])
            input_scaled = scaler.transform(input_data)
            
            lr_prediction = lr_model.predict(input_scaled)[0]
            rf_prediction = rf_model.predict(input_scaled)[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Linear Regression Predicted Profit", f"${lr_prediction:.2f}")
            
            with col2:
                st.metric("Random Forest Predicted Profit", f"${rf_prediction:.2f}")
            
            # Show prediction explanation
            st.markdown("### Prediction Explanation")
            
            if rf_prediction > 0:
                st.success(f"The transaction is predicted to be profitable with an estimated profit of ${rf_prediction:.2f}.")
                
                if discount_input > 20:
                    st.warning("Note: The high discount rate might be reducing your potential profit.")
            else:
                st.error(f"The transaction is predicted to result in a loss of ${abs(rf_prediction):.2f}.")
                
                if discount_input > 20:
                    st.info("Consider reducing the discount to improve profitability.")
                    
                if delivery_input > 5:
                    st.info("Faster delivery time might help improve customer satisfaction and reduce costs.")

except FileNotFoundError:
    st.error("Error: 'Sales.csv' file not found. Please make sure the file is in the same directory as this script.")
    st.markdown("""
    ## How to use this dashboard:
    
    1. Upload the 'Sales.csv' file to the same directory as this script
    2. Restart the Streamlit app
    
    The dashboard will automatically load the data and display all visualizations.
    """)