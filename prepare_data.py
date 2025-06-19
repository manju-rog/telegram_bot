import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

print("Fetching and splitting dataset...")

# Fetch the dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Split the data into an initial set (80%) and a "new" set (20%)
df_initial, df_new = train_test_split(df, test_size=0.2, random_state=42)

# Save the files to the data directory
df_initial.to_csv("housing_initial.csv", index=False)
df_new.to_csv("housing_new_data.csv", index=False)

print("Data preparation complete.")
print(f"Initial data saved to data/housing_initial.csv ({len(df_initial)} rows)")
print(f"New data for retraining saved to data/housing_new_data.csv ({len(df_new)} rows)")