import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the trained model and feature names
model = pickle.load(open("demand_forecast_model.pkl", "rb"))
with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Dummy current stock data and historical trends for demonstration
current_stock = {
    "Pizza": 20,
    "Burger": 35,
    "Salad": 10,
    "Pasta": 15,
    "Dumplings": 5
}
historical_demand = {
    "Pizza": [18, 19, 20, 22, 23],
    "Burger": [30, 32, 33, 34, 36],
    "Salad": [12, 13, 11, 10, 15],
    "Pasta": [14, 13, 15, 16, 15],
    "Dumplings": [4, 6, 7, 5, 5]
}

# Create the main window
window = tk.Tk()
window.title("Restaurant Inventory Demand Forecasting")
window.geometry("1200x800")

# Add a background image
bg_image = Image.open("restaurant.jpg")  # Replace with your image file
bg_image = bg_image.resize((1200, 800))  # Resize to fit the window
bg_photo = ImageTk.PhotoImage(bg_image)
background_label = tk.Label(window, image=bg_photo)
background_label.place(relwidth=1, relheight=1)

# Add a scrollable canvas
canvas = tk.Canvas(window, bg="white", highlightthickness=0)
scrollbar = tk.Scrollbar(window, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg="white")

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="n")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Add a header
header_frame = tk.Frame(scrollable_frame, bg="darkblue", bd=5)
header_frame.pack(pady=20, fill="x")

header_label = tk.Label(
    header_frame,
    text="Restaurant Inventory Demand Forecasting",
    font=("Arial", 20, "bold"),
    fg="white",
    bg="darkblue"
)
header_label.pack()

# Create an input frame
input_frame = tk.Frame(scrollable_frame, bg="white", bd=5)
input_frame.pack(pady=20, fill="x")

# Input fields and labels
tk.Label(input_frame, text="Day of Week (e.g., Monday):", font=("Arial", 14), bg="white").grid(row=0, column=0, pady=10, sticky="e")
day_of_week_entry = tk.Entry(input_frame, font=("Arial", 14))
day_of_week_entry.grid(row=0, column=1, pady=10, padx=10)

tk.Label(input_frame, text="Weather (e.g., Sunny):", font=("Arial", 14), bg="white").grid(row=1, column=0, pady=10, sticky="e")
weather_entry = tk.Entry(input_frame, font=("Arial", 14))
weather_entry.grid(row=1, column=1, pady=10, padx=10)

tk.Label(input_frame, text="Holiday (1=Yes, 0=No):", font=("Arial", 14), bg="white").grid(row=2, column=0, pady=10, sticky="e")
holiday_entry = tk.Entry(input_frame, font=("Arial", 14))
holiday_entry.grid(row=2, column=1, pady=10, padx=10)

# Add predict button
predict_button = tk.Button(
    input_frame,
    text="Predict",
    font=("Arial", 16, "bold"),
    bg="green",
    fg="white",
    command=lambda: predict_demand(scrollable_frame)
)
predict_button.grid(row=3, column=0, columnspan=2, pady=20)

# Create output frame
output_frame = tk.Frame(scrollable_frame, bg="white", bd=5)
output_frame.pack(pady=20, fill="x")

# Table for showing predictions
tree = ttk.Treeview(output_frame, columns=("Food Item", "Current Stock", "Predicted Demand", "Status"), show="headings")
tree.heading("Food Item", text="Food Item")
tree.heading("Current Stock", text="Current Stock")
tree.heading("Predicted Demand", text="Predicted Demand")
tree.heading("Status", text="Status")
tree.pack(fill="both", expand=True)

# Function for demand prediction
def predict_demand(container):
    try:
        # Collect and validate inputs
        day_of_week = day_of_week_entry.get().strip()
        weather = weather_entry.get().strip()
        holiday = holiday_entry.get().strip()

        if not day_of_week or not weather or not holiday.isdigit():
            messagebox.showerror("Invalid Input", "Please fill all fields correctly.")
            return

        holiday = int(holiday)

        # Create DataFrame for input
        input_data = pd.DataFrame({
            'Day_of_Week': [day_of_week],
            'Weather': [weather],
            'Holiday': [holiday]
        })

        # Perform one-hot encoding
        input_data = pd.get_dummies(input_data, columns=['Day_of_Week', 'Weather'])

        # Align with feature names
        for col in feature_names:
            if col not in input_data.columns:
                input_data[col] = 0  # Add missing columns

        input_data = input_data[feature_names]

        # Predict demand
        prediction = model.predict(input_data)
        predicted_demand = int(prediction[0])

        # Clear previous table data
        for i in tree.get_children():
            tree.delete(i)

        # Recommendations and table filling
        for item, stock in current_stock.items():
            status = "Adequate"
            if predicted_demand > stock:
                status = "Restock"
            elif stock >= predicted_demand + 5:
                status = "Available"
            # Insert into table
            tree.insert("", "end", values=(item, stock, predicted_demand, status))

        # Generate bar chart
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(current_stock.keys(), current_stock.values(), label="Current Stock")
        ax.bar(current_stock.keys(), [predicted_demand] * len(current_stock), alpha=0.5, label="Predicted Demand")
        ax.set_ylabel("Quantity")
        ax.set_title("Current Stock vs Predicted Demand")
        ax.legend()

        # Display chart in Tkinter
        canvas_chart = FigureCanvasTkAgg(fig, container)
        canvas_chart.get_tk_widget().pack(pady=20)
        canvas_chart.draw()

        # Generate line chart for historical trends
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        for item, demand in historical_demand.items():
            ax2.plot(range(len(demand)), demand, label=item)
        ax2.set_ylabel("Demand")
        ax2.set_title("Historical Demand Trends")
        ax2.legend()

        # Display line chart in Tkinter
        canvas_chart2 = FigureCanvasTkAgg(fig2, container)
        canvas_chart2.get_tk_widget().pack(pady=20)
        canvas_chart2.draw()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Run the application
window.mainloop()
