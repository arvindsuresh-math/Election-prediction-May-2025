import pandas as pd

# Read the two CSV files
general_df = pd.read_csv('/Users/arvindsuresh/Documents/Github/Election-prediction-May-2025/general_codes.csv')
race_df = pd.read_csv('/Users/arvindsuresh/Documents/Github/Election-prediction-May-2025/race_by_sex_codes.csv')

# Concatenate the dataframes
combined_df = pd.concat([general_df, race_df], ignore_index=True)

# Write to new CSV file
combined_df.to_csv('/Users/arvindsuresh/Documents/Github/Election-prediction-May-2025/combined_codes.csv', index=False)

print("Successfully merged the CSV files into combined_codes.csv")
print(f"Total rows: {len(combined_df)}")
print(f"General codes rows: {len(general_df)}")
print(f"Race codes rows: {len(race_df)}")