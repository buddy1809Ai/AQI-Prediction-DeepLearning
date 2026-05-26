import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
df = pd.read_csv("aqi_data.csv")

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Select required columns
df = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'AQI']]

# Remove missing values
df = df.dropna()

# Features and target
X = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']]
y = df['AQI']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Build ANN model
model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(5,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile model
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)

# Train model
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# Predictions
predictions = model.predict(X_test)

# Accuracy
mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", mae)

# Save model
model.save("aqi_model.h5")

print("Model trained successfully!")
