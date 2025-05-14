import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

df = pd.read_csv('Sales.csv')
df['Discount'] = (df['Discount']) * 100

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
        

X = df[['Delivery Time','Quantity', 'Category', 'Sub-Category', 'Discount','Profit']]
y = np.log1p(df['Sales'])  #For Reverse to get actual value --> y_pred_actual = np.expm1(y_pred)

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 


params = {'max_depth': 11, 'n_estimators' : 150}

rf = RandomForestRegressor(**params)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
rf_train = rf.score(X_train, y_train)
rf_test = rf.score(X_test, y_test)

print('Training Accuracy:', rf_train)
print('Testing Accuracy:', rf_test)

with open('rf_model.pkl', 'wb') as model_file:
    pickle.dump(rf, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
