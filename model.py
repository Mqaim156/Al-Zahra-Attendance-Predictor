import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# import the data
df = pd.read_csv("attendance1.csv")

# --- 2. Feature Preprocessing ---

# Drop columns that are just identifiers and won't help predict
if 'event_id' in df.columns:
    df = df.drop('event_id', axis=1)

# Define our features (X) and target (y)
target_variable = 'attendance'
X = df.drop(target_variable, axis=1)
y = df[target_variable]

print("--- Data Head (First 5 Rows of X) ---")
print(X.head())
print("\n")

# --- 3. Time Series Data Split ---
# Your data is chronological. We CANNOT split it randomly.
# We must use the past to predict the future.
# We will use the first 80% of data to train and the last 20% to test.

split_percentage = 0.8
split_point = int(len(df) * split_percentage)

X_train = X.iloc[:split_point]
y_train = y.iloc[:split_point]

X_test = X.iloc[split_point:]
y_test = y.iloc[split_point:]

print(f"Total samples: {len(df)}")
print(f"Training samples: {len(X_train)} (First {split_point} rows)")
print(f"Testing samples: {len(X_test)} (Last {len(X_test)} rows)")
print("\n")

# --- 4. Initialize and Train Ridge Regression ---

# A. Define which columns are categorical for this model
#    'program_type' is the ONLY column that needs encoding.
categorical_features = ['program_type']

# B. Create a preprocessor
# We must use One-Hot Encoding for linear models.
# 'handle_unknown='ignore'' prevents errors if a new category appears
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    # 'remainder=passthrough' will automatically find all other columns
    # (is_summer, is_weekend, etc.) and pass them through untouched,
    # which is exactly what we want.
    remainder='passthrough'  
)

# C. Apply the preprocessing to our data
# NOTE: We fit on TRAIN data, then transform BOTH train and test
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# D. Create and train the Ridge Regression model
model = Ridge(alpha=0.5)  # alpha is the regularization strength

print("--- Training Ridge Regression Model ---")
model.fit(X_train_processed, y_train)
print("Model training complete.")
print("\n")

model_filename = 'attendance_model.joblib'
preprocessor_filename = 'preprocessor.joblib'

# --- 4.5 Saving the Model and Preprocessor with Joblib--- 
# Save the trained model
joblib.dump(model, model_filename)
print(f"Trained model saved toL {model_filename}")

# Save the preprocessor
joblib.dump(preprocessor, preprocessor_filename)
print(f"Preprocessor saved to: {preprocessor_filename}")
print("\n")

# --- 5. predict testing data ---
y_pred = model.predict(X_test_processed)

# Calculate error metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Display the results
print("--- Model Evaluation Metrics ---")
print(f"  Mean Absolute Pct Error (MAPE): {mape:.2%}")
print("--- Model Evaluation on Test Set ---")
print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"  Mean Absolute Error (MAE):    {mae:.2f}")
print(f"  R-squared (R²):               {r2:.2f}")
print("\n")

# --- Interpretation of Metrics ---
print("--- How to Read These Metrics ---")
print(f"> **MAE**: On average, the model's prediction was off by ~{mae:.0f} people.")
print(f"> **RMSE**: This is a similar metric to MAE but gives higher penalties for large errors.")
print(f"> **R²**: An R² of 1.0 is a perfect model. An R² of 0.0 means the model is no better than just guessing the average attendance. A negative R² means the model is *worse* than guessing the average.")
print("\n")


# --- 6. Plot Feature Importance (Coefficients) ---
# For a Linear Model (Ridge), "importance" is the coefficient
# value. It shows the positive or negative impact of each feature.

print("--- Displaying Feature Importance (Model Coefficients) ---")
print("This chart shows how much each factor (positively or negatively) affects the attendance prediction.")

# A. Get the coefficients from the trained model
coefficients = model.coef_

# B. Get the feature names from the preprocessor
#    This is CRITICAL - it gets the names for all the
#    one-hot-encoded columns (e.g., 'day_of_week_Mon')
#    and the passthrough columns.
feature_names = preprocessor.get_feature_names_out()

# C. Create a pandas Series to easily match names to values
importance_series = pd.Series(coefficients, index=feature_names)

# D. Sort the values for a cleaner plot
sorted_importance = importance_series.sort_values()

# E. Plot
fig, ax = plt.subplots(figsize=(10, 8)) # Made it a bit taller
sorted_importance.plot(kind='barh', ax=ax)
ax.set_title("Feature Importance (Coefficients) for Ridge Model")
ax.set_xlabel("Coefficient Value (Impact on Attendance)")
plt.tight_layout()
plt.show()


# --- 7. Show Sample Predictions ---
# Let's see what the model predicted vs. what actually happened
# for the last few events.
comparison_df = pd.DataFrame({
    'Actual Attendance': y_test,
    'Predicted Attendance': np.round(y_pred, 0)
})
print("--- Full Predictions vs. Actual on Test Set ---")
print(comparison_df)


# --- 8. PREDICTING A NEW EVENT ---

# 1. Create a new event as a DataFrame
#    You MUST use the exact same column names your model was trained on
#    (before preprocessing)
new_event_data = {
    'is_summer': [0],               # 1 for Yes, 0 for No
    'is_weekend': [0],              # 1 for Yes, 0 for No
    'program_type': ['small'],   # Must be one of the categories it has seen
    'special_speaker_flag': [0],    # 1 for Yes, 0 for No
    'is_special_month': [0],        # 1 for Yes, 0 for No
    'is_food': [0]                  # 1 for Yes, 0 for No
}

# Convert this dictionary into a DataFrame
# (The [0] list format is required for a single row)
new_event_df = pd.DataFrame(new_event_data)

print("\n--- New Event to Predict ---")
print(new_event_df)


# 2. Transform the new data
#    USE THE PREPROCESSOR YOU ALREADY FITTED.
#    We call .transform() NOT .fit_transform()
new_event_processed = preprocessor.transform(new_event_df)


# 3. Make the prediction
predicted_attendance = model.predict(new_event_processed)

# The output is an array, so we get the first (and only) item
prediction = predicted_attendance[0]


print("\n--- Prediction Result ---")
print(f"Predicted Attendance: {prediction:.0f} people")