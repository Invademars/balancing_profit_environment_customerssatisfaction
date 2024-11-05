import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# First, let's recreate the dummy data for training (needed for scaler)
np.random.seed(42)
n_samples = 1000

profit_margin = np.random.uniform(5, 25, n_samples)
co2_emissions = np.random.uniform(1000, 5000, n_samples)
solid_waste = np.random.uniform(100, 500, n_samples)
energy_consumption = np.random.uniform(500000, 2000000, n_samples)

customer_satisfaction = (
    10 - (profit_margin - 5) * 0.1
    - (co2_emissions - 1000) * 0.001
    - (solid_waste - 100) * 0.005
    - (energy_consumption - 500000) * 0.000002
    + np.random.normal(0, 0.5, n_samples)
)
customer_satisfaction = np.clip(customer_satisfaction, 0, 10)

# Create DataFrame and scaler
data = pd.DataFrame({
    'profit_margin': profit_margin,
    'co2_emissions': co2_emissions,
    'solid_waste': solid_waste,
    'energy_consumption': energy_consumption
})

scaler = StandardScaler()
scaler.fit(data)

# Create and train the model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(4,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(scaler.transform(data), customer_satisfaction, epochs=100, verbose=0)

def predict_satisfaction_interactive():
    print("\n=== Customer Satisfaction Predictor ===")
    print("\nPlease enter the following values:")

    try:
        # Get input values with value ranges
        profit = float(input("\nProfit Margin (5-25%): "))
        co2 = float(input("CO2 Emissions (1000-5000 tons): "))
        waste = float(input("Solid Waste (100-500 tons): "))
        energy = float(input("Non-clean Energy Consumption (500000-2000000 kWh): "))

        # Create input array
        input_data = np.array([[profit, co2, waste, energy]])

        # Scale the input
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)[0][0]
        prediction = np.clip(prediction, 0, 10)

        print("\n=== Results ===")
        print(f"\nPredicted Customer Satisfaction: {prediction:.2f}/10")

        # Provide interpretation
        if prediction >= 8:
            print("Interpretation: Excellent! Customers are likely to be very satisfied.")
        elif prediction >= 6:
            print("Interpretation: Good. Customers are likely to be satisfied.")
        elif prediction >= 4:
            print("Interpretation: Fair. There's room for improvement.")
        else:
            print("Interpretation: Poor. Consider adjusting the parameters.")

        # Suggestions for improvement
        print("\nSuggestions:")
        if profit > 20:
            print("- Consider reducing profit margin slightly")
        if co2 > 3000:
            print("- Look for ways to reduce CO2 emissions")
        if waste > 300:
            print("- Implement better waste management practices")
        if energy > 1500000:
            print("- Consider incorporating more clean energy sources")

    except ValueError:
        print("\nError: Please enter numeric values only.")
        return

    # Ask if user wants to try another prediction
    again = input("\nWould you like to try another prediction? (yes/no): ")
    if again.lower() in ['yes', 'y']:
        predict_satisfaction_interactive()

# Function to run multiple scenarios at once
def predict_multiple_scenarios():
    scenarios = []
    print("\n=== Multiple Scenarios Predictor ===")

    try:
        num_scenarios = int(input("\nHow many scenarios would you like to compare? "))

        for i in range(num_scenarios):
            print(f"\nScenario {i+1}:")
            profit = float(input("Profit Margin (5-25%): "))
            co2 = float(input("CO2 Emissions (1000-5000 tons): "))
            waste = float(input("Solid Waste (100-500 tons): "))
            energy = float(input("Non-clean Energy Consumption (500000-2000000 kWh): "))

            scenarios.append({
                'profit_margin': profit,
                'co2_emissions': co2,
                'solid_waste': waste,
                'energy_consumption': energy
            })

        print("\n=== Comparison Results ===")
        print("\nScenario | Satisfaction | Details")
        print("-" * 50)

        for i, scenario in enumerate(scenarios, 1):
            input_data = np.array([[
                scenario['profit_margin'],
                scenario['co2_emissions'],
                scenario['solid_waste'],
                scenario['energy_consumption']
            ]])

            prediction = model.predict(scaler.transform(input_data))[0][0]
            prediction = np.clip(prediction, 0, 10)

            print(f"   {i}     |    {prediction:.2f}/10   | Profit: {scenario['profit_margin']}%, "
                  f"CO2: {scenario['co2_emissions']}t")

    except ValueError:
        print("\nError: Please enter numeric values only.")
        return

# Main menu
def main_menu():
    while True:
        print("\n=== Main Menu ===")
        print("1. Single Scenario Prediction")
        print("2. Compare Multiple Scenarios")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ")

        if choice == '1':
            predict_satisfaction_interactive()
        elif choice == '2':
            predict_multiple_scenarios()
        elif choice == '3':
            print("\nThank you for using the Customer Satisfaction Predictor!")
            break
        else:
            print("\nInvalid choice. Please try again.")

# Run the program
if __name__ == "__main__":
    print("Welcome to the Construction Project Customer Satisfaction Predictor!")
    main_menu()