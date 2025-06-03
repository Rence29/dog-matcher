import pandas as pd
import re
import os

# --- Configuration ---
DATA_FILE = 'dog_adoption_augmented.csv'
PROCESSED_DATA_FILE = 'dog_adoption_preprocessed.csv'
ENCODED_FEATURES_FILE = 'encoded_features.txt' # To save the list of features after encoding

# --- 1. Load the dataset ---
print(f"Loading data from {DATA_FILE}...")
if not os.path.exists(DATA_FILE):
    print(f"Error: {DATA_FILE} not found. Please ensure your dataset is in the same directory.")
    exit()

df = pd.read_csv(DATA_FILE)
print("Data loaded successfully.")
print("\nOriginal Data Info:")
df.info()
print("\nFirst 5 rows of original data:")
print(df.head())

# --- 2. Preprocess 'Dog_Age' column ---
print("\nPreprocessing 'Dog_Age'...")
def convert_dog_age(age_str):
    if 'years' in age_str:
        return float(age_str.replace(' years', ''))
    elif 'year' in age_str: # Handle singular 'year'
        return float(age_str.replace(' year', ''))
    elif 'months' in age_str: # Convert months to years
        return float(age_str.replace(' months', '')) / 12
    elif 'month' in age_str: # Handle singular 'month'
        return float(age_str.replace(' month', '')) / 12
    else:
        # Try to convert directly if no unit is found, assuming it's already a number
        try:
            return float(age_str)
        except ValueError:
            return None # Or handle as appropriate, e.g., median imputation later

df['Dog_Age_Years'] = df['Dog_Age'].apply(convert_dog_age)
df.drop('Dog_Age', axis=1, inplace=True) # Remove original 'Dog_Age' column
print(" 'Dog_Age' converted to numerical 'Dog_Age_Years'.")


# --- 3. Preprocess 'Match_Category' (Target Variable) ---
print("Preprocessing 'Match_Category'...")
# Map 'Good Match ✅' to 1 and 'Bad Match ❌' to 0
df['Match_Outcome'] = df['Match_Category'].map({'Good Match ✅': 1, 'Bad Match ❌': 0})
df.drop('Match_Category', axis=1, inplace=True) # Remove original 'Match_Category' column
print(" 'Match_Category' converted to numerical 'Match_Outcome'.")


# --- 4. Handle Categorical Features (One-Hot Encoding) ---
print("\nPerforming One-Hot Encoding for categorical features...")
categorical_cols = [
    'Breed', 'Behavior', 'Size', 'Health_Condition',
    'Housing_Type', 'Lifestyle', 'Family_Composition', 'Pet_Experience'
]

# Create dummy variables (one-hot encoding)
# `drop_first=True` avoids multicollinearity, which is good practice.
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("One-Hot Encoding complete.")
print("\nFirst 5 rows of preprocessed data:")
print(df_encoded.head())
print("\nPreprocessed Data Info:")
df_encoded.info()

# --- 5. Save the preprocessed data and feature list ---
print(f"\nSaving preprocessed data to {PROCESSED_DATA_FILE}...")
df_encoded.to_csv(PROCESSED_DATA_FILE, index=False)
print("Preprocessed data saved.")

# Save the list of feature columns for later use in the API
features_list = df_encoded.drop('Match_Outcome', axis=1).columns.tolist()
with open(ENCODED_FEATURES_FILE, 'w') as f:
    for item in features_list:
        f.write("%s\n" % item)
print(f"List of encoded features saved to {ENCODED_FEATURES_FILE}.")

print("\nData preprocessing complete. You can now proceed to train your model.")