import tkinter as tk
from tkinter import messagebox
import numpy as np
import pickle

# === Load Model and Scaler ===
with open('rf_model.pkl', 'rb') as model_file:
    rf = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# === Main Window Setup ===
root = tk.Tk()
root.title("Sales Predictor")
root.geometry("850x600")
root.configure(bg="#f0f8ff")  # Light blue background

# === Title ===
title = tk.Label(root, text="🔮 Smart Sales Prediction", font=("Segoe UI", 22, "bold"), bg="#f0f8ff", fg="#0f1f19")
title.pack(pady=20)

# === Main Container Frame ===
main_frame = tk.Frame(root, bg="#e0f7fa", bd=3, relief="groove")
main_frame.pack(padx=30, pady=10, fill="x", expand=False)

# Configure grid for button centering
main_frame.grid_rowconfigure(0, weight=1)
main_frame.grid_columnconfigure(0, weight=1)
main_frame.grid_columnconfigure(1, weight=1)

# === Left Input Frame ===
input_frame = tk.Frame(main_frame, bg="#e0f7fa")
input_frame.grid(row=0, column=0, padx=20, pady=10, sticky="nw")

# === Right Button Frame ===
button_frame = tk.Frame(main_frame, bg="#e0f7fa")
button_frame.grid(row=0, column=1, padx=20, pady=10, sticky="nsew")
button_frame.grid_rowconfigure(0, weight=1)
button_frame.grid_columnconfigure(0, weight=1)

# === Input Fields ===
labels = [
    ("Delivery Time", "4"),
    ("Quantity", "2"),
    ("Category (encoded)", "1"),
    ("Sub-Category (encoded)", "3"),
    ("Discount (%)", "10.0"),
    ("Profit ($)", "50.0"),
]

entries = []

for i, (label_text, default) in enumerate(labels):
    tk.Label(input_frame, text=label_text, font=("Segoe UI", 12), bg="#e0f7fa").grid(row=i, column=0, sticky="w", pady=5)
    entry = tk.Entry(input_frame, font=("Segoe UI", 12), width=28)
    entry.insert(0, default)
    entry.grid(row=i, column=1, pady=5, padx=10)
    entries.append(entry)

# === Predict Button (centered vertically) ===
def predict():
    try:
        values = [float(entry.get()) for entry in entries]
        values = np.array([values])
        scaled = scaler.transform(values)
        log_pred = rf.predict(scaled)
        actual = np.expm1(log_pred)[0]
        result_label.config(text=f"💰 Predicted Sales: ${actual:,.2f}")
    except Exception as e:
        messagebox.showerror("Input Error", f"Please enter valid numbers.\n\n{e}")

predict_btn = tk.Button(button_frame, text="🧠 Predict\nNow", font=("Segoe UI", 14, "bold"),
                        bg="#00b894", fg="white", activebackground="#55efc4",
                        padx=20, pady=20, width=12, height=3, command=predict)
predict_btn.grid(row=0, column=0)

# === Result Display Frame ===
result_frame = tk.Frame(root, bg="#ffeaa7", bd=2, relief="solid")
result_frame.pack(pady=20, padx=30, fill="x")

result_label = tk.Label(result_frame, text="💰 Predicted Sales: —", font=("Segoe UI", 16, "bold"), bg="#ffeaa7", fg="#2d3436")
result_label.pack(pady=15)

# === Title Animation ===
# def fade_in(widget, step=5):
#     r = int(55 + step)
#     g = int(85 + step)
#     b = int(123 + step)
#     color = f"#{r:02x}{g:02x}{b:02x}"
#     widget.configure(fg=color)
#     if step < 100:
#         root.after(10, lambda: fade_in(widget, step + 1))

# fade_in(title)

# === Launch App ===
root.mainloop()
