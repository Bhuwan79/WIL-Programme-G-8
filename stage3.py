import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Load scaled data
X_train_scaled = pd.read_csv('Data_Preparation/X_train_scaled.csv')
X_test_scaled = pd.read_csv('Data_Preparation/X_test_scaled.csv')
y_train = pd.read_csv('Data_Preparation/y_train.csv')
y_test = pd.read_csv('Data_Preparation/y_test.csv')

# Convert y_train and y_test to 1D arrays
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Define ANN model
model = Sequential()

# Input layer and first hidden layer
model.add(Dense(units=64, activation='relu', input_shape=(X_train_scaled.shape[1],)))

# Second hidden layer
model.add(Dense(units=32, activation='relu'))

# Output layer
model.add(Dense(units=1, activation='sigmoid'))  # For binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=50, callbacks=[early_stopping])

# Save the model
model.save('Churn_Prediction_Model/ann_model.h5')


#####Evaluate the Model##############

# Predict on test set
y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification report
report = classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])
print(report)

# Save the report
with open('Churn_Prediction_Model/classification_report.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy:.2f}\n\n")
    f.write(report)


