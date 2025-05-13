import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
import pickle

df = pd.read_csv('Sales.csv')
df['Discount'] = (df['Discount']) * 100
df['Product_Popularity'] = df.groupby('Product ID')['Order ID'].transform('count')
df['Customer_Order_Count'] = df.groupby('Customer ID')['Order ID'].cumcount() + 1

df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

df['Delivery Time'] = (df['Ship Date'] - df['Order Date']).dt.days
df['Delivery Time'].value_counts()


# Columns to encode (useful categorical features)
cols_to_encode = [
    'Ship Mode',      
    'Segment',
    'City',           
    'State',          
    'Region',         
    'Category',       
    'Sub-Category',   
]

# Apply LabelEncoder to each column
le = LabelEncoder()
for col in cols_to_encode:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])
        

X = df[['Delivery Time','Quantity', 'Category', 'Sub-Category', 'Discount','Profit','Product_Popularity','Customer_Order_Count']]
y = np.log1p(df['Sales'])  #For Reverse to get actual value --> y_pred_actual = np.expm1(y_pred)

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 


params = {
    'n_estimators': 250,
    'learning_rate': 0.03,      
    'max_depth': 8,
    'min_child_weight': 6,      
    'gamma': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.5,
    'reg_lambda': 5
}

model = XGBRegressor(**params)
model.fit(X_train, y_train)

print('Train:',model.score(X_train, y_train))
print('Test:',model.score(X_test, y_test))

with open('xgb_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
