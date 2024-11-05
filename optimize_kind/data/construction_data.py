import numpy as np
import pandas as pd
import os

# Print current working directory
current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")

# Create 'data' folder if it doesn't exist
data_folder = os.path.join(current_directory, 'construction_data')
os.makedirs(data_folder, exist_ok=True)

# Generate data (same as before)
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

# Create DataFrame
data = pd.DataFrame({
    'profit_margin': profit_margin,
    'co2_emissions': co2_emissions,
    'solid_waste': solid_waste,
    'energy_consumption': energy_consumption,
    'customer_satisfaction': customer_satisfaction
})

# Create description
data_description = pd.DataFrame({
    'min': data.min(),
    'max': data.max(),
    'mean': data.mean(),
    'std': data.std()
})

# Save files with full path
data_file_path = os.path.join(data_folder, 'construction_data.csv')
description_file_path = os.path.join(data_folder, 'construction_data_description.csv')

# Save files
data.to_csv(data_file_path, index=False)
data_description.to_csv(description_file_path)

# Print file locations
print(f"\nFiles have been saved at:")
print(f"1. Main data: {data_file_path}")
print(f"2. Description: {description_file_path}")

# Print first few rows
print("\nFirst few rows of the dataset:")
print(data.head())

# Print description
print("\nDataset Description:")
print(data_description)