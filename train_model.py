import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# --- Configuration ---
PROCESSED_DATA_FILE = 'dog_adoption_preprocessed.csv'
MODEL_FILE = 'random_forest_dog_matcher_model.pkl'
ENCODED_FEATURES_FILE = 'encoded_features.txt' # To load the order of features

# --- 1. Load Preprocessed Data ---
print(f"Loading preprocessed data from {PROCESSED_DATA_FILE}...")
if not os.path.exists(PROCESSED_DATA_FILE):
    print(f"Error: {PROCESSED_DATA_FILE} not found. Please run preprocess_data.py first.")
    exit()

try:
    df = pd.read_csv(PROCESSED_DATA_FILE)
    print("Preprocessed data loaded successfully.")
    print("\nPreprocessed Data Head:")
    print(df.head())
except Exception as e:
    print(f"Error loading preprocessed data: {e}")
    exit()

# --- 2. Define Features (X) and Target (y) ---
# The target variable is 'Match_Outcome'. All other columns are features.
features = [col for col in df.columns if col != 'Match_Outcome']
X = df[features]
y = df['Match_Outcome']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Number of features: {len(features)}")


# --- 3. Split Data into Training and Testing Sets ---
# We split the data to evaluate the model's performance on unseen data.
# 80% for training, 20% for testing. `random_state` ensures reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# `stratify=y` ensures that the train and test sets have roughly the same proportion of target labels as the input dataset.

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")


# --- 4. Initialize and Train the Random Forest Classifier ---
# n_estimators: The number of decision trees in the forest. More trees generally means better performance,
#               but also longer training time and higher memory usage. 100 is a good starting point.
# random_state: Ensures that you get the same results every time you run the script.
print("\nTraining Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
# `class_weight='balanced'` helps if you have an imbalanced dataset (e.g., more 'Good Match' than 'Bad Match').
model.fit(X_train, y_train)
print("Model training complete.")


# --- 5. Evaluate the Model (Important for understanding performance) ---
print("\nEvaluating model performance on the test set...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Bad Match', 'Good Match'], zero_division=0)

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)

if accuracy < 0.75:
    print("\n--- WARNING ---")
    print("Model accuracy is relatively low. Consider:")
    print("1. Collecting more diverse and balanced data.")
    print("2. Trying different features or more advanced preprocessing.")
    print("3. Tuning Random Forest hyperparameters (e.g., n_estimators, max_depth).")
    print("-----------------")

# --- 6. Save the Trained Model ---
# This saves the trained model to a file, so you don't need to retrain it every time your API starts.
joblib.dump(model, MODEL_FILE)
print(f"\nModel saved successfully to {MODEL_FILE}")

print("Training script finished.")