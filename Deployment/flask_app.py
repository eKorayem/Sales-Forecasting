from flask import Flask, render_template, request
import numpy as np
import pickle

# Load model and scaler
with open('xgb_model.pkl', 'rb') as model_file:
    xgb = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    scroll_to_result = False

    default_values = {
        "delivery_time": 4,
        "quantity": 2,
        "category": 1,
        "sub_category": 3,
        "discount": "15.0",
        "profit": "50.0",
        "popularity": 8,
        "order_count": 2
    }

    if request.method == 'POST':
        try:
            delivery_time = float(request.form['delivery_time'])
            quantity = int(request.form['quantity'])
            category = int(request.form['category'])
            sub_category = int(request.form['sub_category'])
            discount = float(request.form['discount'])
            profit = float(request.form['profit'])
            popularity = int(request.form['popularity'])
            order_count = int(request.form['order_count'])

            features = np.array([[delivery_time, quantity, category, sub_category,
                                  discount, profit, popularity, order_count]])
            features_scaled = scaler.transform(features)
            log_sales = xgb.predict(features_scaled)
            actual_sales = np.expm1(log_sales)[0]
            prediction = f"${actual_sales:.2f}"
            scroll_to_result = True

            default_values.update({
                "delivery_time": delivery_time,
                "quantity": quantity,
                "category": category,
                "sub_category": sub_category,
                "discount": discount,
                "profit": profit,
                "popularity": popularity,
                "order_count": order_count
            })

        except ValueError:
            error = "Please enter valid numerical values."
            scroll_to_result = True

    return render_template('index.html', prediction=prediction, error=error, defaults=default_values, anchor=scroll_to_result)

if __name__ == '__main__':
    app.run(debug=True)
