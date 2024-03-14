# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 00:29:06 2023

@author: cleme
"""

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the data
file_path = 'clean_data_v2.xlsx'
df = pd.read_excel(file_path)

# Check unique values in the "days difference" column
unique_values = df["days difference"].astype(str).str.strip().str.lower().unique()
#print("Unique values in 'days difference' column:", unique_values)

# Convert "days difference" to numeric, ignoring errors to handle non-numeric values
df["days difference"] = pd.to_numeric(df["days difference"], errors='coerce')

# Create a new column "category" based on the specified criteria
df['category'] = pd.cut(df['days difference'],
                        bins=[-np.inf, 125, 250, np.inf],
                        labels=['short', 'medium', 'long'])

# For 'unknown' or NaN values, assign the 'unknown' label
df['category'] = df['category'].cat.add_categories(['unknown']).fillna('unknown')

# Clean up column names by removing leading and trailing whitespaces
df.columns = df.columns.str.strip()

# Specify the desired number of samples for each category
desired_short_samples = 28
desired_medium_samples = 18
desired_long_samples = 17
desired_unknown_samples = 8

# Sample the desired number of samples for each category
short_samples = df[df['category'] == 'short'].sample(n=desired_short_samples)
medium_samples = df[df['category'] == 'medium'].sample(n=desired_medium_samples)
long_samples = df[df['category'] == 'long'].sample(n=desired_long_samples)
unknown_samples = df[df['category'] == 'unknown'].sample(n=desired_unknown_samples)

# Combine the samples to create the training dataset
training_data = pd.concat([short_samples, medium_samples, long_samples, unknown_samples])

# Shuffle the training dataset
training_data = training_data.sample(frac=1).reset_index(drop=True)

# Use the remaining data as the test dataset
test_data = df.drop(training_data.index)

# Features (X) and target variable (y)
X_train = training_data.drop(['days difference', 'category'], axis=1)  # Features
y_train = training_data['category']  # Target variable

# Split the test data into features (X_test) and target variable (y_test)
X_test = test_data.drop(['days difference', 'category'], axis=1)  # Features
y_test = test_data['category']  # Target variable

# Initialize the RandomForestClassifier
rf_classifier = RandomForestClassifier()

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['short', 'medium', 'long', 'unknown'], yticklabels=['short', 'medium', 'long', 'unknown'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Create a DataFrame to show feature importances
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print or visualize the feature importances
print(feature_importance_df)
print("The length of train X is: ", len(X_train))
print("The length of train y is: ", len(y_train))
print("The length of test X is: ", len(X_test))
print("The length of test y is: ", len(y_test))