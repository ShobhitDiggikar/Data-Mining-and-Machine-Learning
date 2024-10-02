#!/usr/bin/env python
# coding: utf-8

# # Outline
# - 1.problem Statement
# - 2.Why is it important
# - 3.key Stakeholders
# - 4.Data Preprocessing
# - 5.Exploratory data analysis
# - 6.Models
# - 7.Conclusion
# - 8.My Thoughts on assumptions

# # 1 Problem Statement
# The main objective is to create a predictive model that effectively anticipates cancellations of hotel bookings. Leveraging the given dataset, which encompasses diverse factors like lead time, hotel type, customer particulars, and booking details, the model should demonstrate the capability to forecast whether a specific reservation will be canceled.

# # Why is it important
# - 1. Enhancing Operational Effectiveness: Anticipating cancellations in advance enables hotels to streamline inventory management, allocate resources more effectively, and potentially sell unused rooms.
# - 2. Optimizing Revenue: Precise predictions contribute to minimizing losses attributed to cancellations and refining pricing strategies for revenue optimization.
# - 3. Gaining Customer Insights: Insight into cancellation patterns and reasons offers valuable information about customer behavior, facilitating targeted marketing efforts and enhanced customer service.
# - 4. Securing Competitive Edge: In a fiercely competitive industry, the ability to foresee and respond to booking trends stands as a substantial advantage.

# # Key Stakeholders
# - 1. Hotel Administration: Seeks valuable insights for strategic decisions regarding room assignments, pricing tactics, and overall managerial actions.
# - 2. Front Desk and Reservation Staff: Engaged in direct booking processes and customer interactions, requiring information for effective operational adjustments.
# - 3. Marketing and Sales Divisions: Utilize insights to comprehend customer behavior and tailor marketing approaches to mitigate cancellations.
# - 4. Financial Department: Concerned with forecasts, budgeting, and financial planning, making accurate predictions crucial.
# - 5. Guests: Indirectly impacted, as enhanced predictions contribute to improved services and potentially more competitive pricing.
# - 6. IT/Data Science Team: Tasked with developing and maintaining the predictive model, ensuring its precision and ongoing relevance.

# # Data Preprocessing 

# In[3]:


import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


data = pd.read_csv('/Users/shobhitdhanyakumardiggikar/Downloads/hotel_bookings.csv')  # Read the CSV file into a DataFrame
print("Data loaded from CSV:\n")
print(data.head())  # Display the first 5 rows of dataset
print("\n")

# Eliminating two columns which are 'agent' and 'company' columns because of too many null values
data.drop(['agent', 'company'], axis=1, inplace=True)
print("Data after dropping 'agent' and 'company' columns:\n")
print(data.head())
print("\n")

# Eliminating the rows with null values
data.dropna(inplace=True)
print("Data after removing rows with null values:\n")
print(data.head())
print("\n")

# Store the initial number of rows for later comparison
initial_row_count = data.shape[0]
# Eliminate duplicate rows to avoid redundancy
data.drop_duplicates(inplace=True)
print(f"Removed {initial_row_count - data.shape[0]} duplicate rows.\n")

# Store the initial number of columns for later comparison
initial_column_count = data.shape[1]

# Find columns with data type 'object' which typically indicates categorical data
categorical_columns = data.select_dtypes(include=['object']).columns

# Initialize a list to keep track of new columns added after one-hot encoding
new_columns = []
# Iterate over each categorical column for one-hot encoding
for column in categorical_columns:
    print(f"One-hot encoding the '{column}' column...")
    dummies = pd.get_dummies(data[column], prefix=column)
    new_columns.extend(dummies.columns)
    data = pd.concat([data, dummies], axis=1)
    data.drop(column, axis=1, inplace=True)
    print(f"Finished one-hot encoding the '{column}' column.\n")

# Find boolean columns in the DataFrame
boolean_columns = data.select_dtypes(include=['bool']).columns
for column in boolean_columns:
    print(f"Converting boolean column '{column}' to binary (1/0)...")
    data[column] = data[column].astype(int)
    print(f"Finished converting boolean column '{column}' to binary (1/0).\n")

# Finding numerical columns for outlier detection
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
for column in numerical_columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    print(f"Removed {outliers.shape[0]} outliers from the '{column}' column.\n")




# In[5]:


data.info


# # Exploratory Data Analysis

# In[39]:


data.isnull().all()


# In[41]:


data.describe()


# In[36]:


data.hist(figsize=(70,80))
plt.show()


# # train-test-split

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your data
# data = pd.read_csv('your_data_file.csv')  # Uncomment this line and specify your data file

# Assuming 'is_canceled' is the target variable for a classification problem
y = data['is_canceled']
X = data.drop('is_canceled', axis=1)

# Feature selection using a tree-based estimator
model = ExtraTreesClassifier(n_estimators=100, random_state=42)  # Added random_state for reproducibility
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# Plot the feature importances with different colors and shades
top_n = 20
plt.figure(figsize=(10, 10))
colors = sns.color_palette(['blue', 'lightcoral', 'lightgreen'])
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(top_n), palette=colors)
plt.title('Top Feature Importances')
plt.tight_layout()
plt.show()

# Selecting the top 'n' features
selected_features = feature_importance_df.head(top_n)['Feature'].tolist()
data_reduced = data[selected_features + ['is_canceled']]

# Normalize the selected features
scaler = StandardScaler()
data_reduced[selected_features] = scaler.fit_transform(data_reduced[selected_features])

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data_reduced.drop('is_canceled', axis=1), data_reduced['is_canceled'], test_size=0.2, random_state=42)

# Print details of the training and test sets
print("\nData has been split into training and test sets:")
print(f"Training set has {X_train.shape[0]} rows and {X_train.shape[1]} features.")
print(f"Test set has {X_test.shape[0]} rows and {X_test.shape[1]} features.")


# # Model Building and Evaluation

# # Logistic Regression

# In[13]:


import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],  # Inverse of regularization strength
    'penalty': ['l1', 'l2'],  # Type of regularization
    'solver': ['liblinear']   # Solver for optimization (liblinear is good for small datasets)
}

# Create a Logistic Regression model with random_state for reproducibility
logreg = LogisticRegression(random_state=42)

# Initialize GridSearchCV with n_jobs=-1 for parallel processing
grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

# Time the fitting process
start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Run-time information
print("Run-time for GridSearchCV fitting: {:.2f} seconds".format(end_time - start_time))

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate and print metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Data requirements (general guideline)
num_features = X_train.shape[1]
print("Number of features:", num_features)
print("Rule of Thumb for Minimum Number of Data Points:", num_features * 10)  # This is a simplistic rule of thumb


# # Random Forest

# In[14]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming you have loaded your dataset into X and y
# X should contain features, and y should contain the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Random Forest model
random_forest_model = RandomForestClassifier(random_state=42)

# Fit the model to the training data
random_forest_model.fit(X_train, y_train)

# Predict on the test data
y_pred = random_forest_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))


# # XGBoost 

# In[16]:


get_ipython().system('pip install xgboost')


# In[17]:


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming you have loaded your dataset into X and y
# X should contain features, and y should contain the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42)

# Fit the model to the training data
xgb_model.fit(X_train, y_train)

# Predict on the test data
y_pred = xgb_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))


# # KNN

# In[19]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Assuming you have loaded your dataset into X and y
# X should contain features, and y should contain the target variable

# Convert the data to NumPy arrays
X_train_np = np.array(X_train)
X_test_np = np.array(X_test)
y_train_np = np.array(y_train)
y_test_np = np.array(y_test)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the KNN model (you can adjust the 'n_neighbors' parameter)
knn_model = KNeighborsClassifier(n_neighbors=5)

# Fit the model to the training data
knn_model.fit(X_train_np, y_train_np)

# Predict on the test data
y_pred = knn_model.predict(X_test_np)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test_np, y_pred))
print("Precision:", precision_score(y_test_np, y_pred))
print("Recall:", recall_score(y_test_np, y_pred))
print("F1 Score:", f1_score(y_test_np, y_pred))


# # The Winning Model is XGBoost

# # Conclusion

# Based on all the 4 models that we have used right now, we can see that "XGBoost" provides us with the best accuracy, that is, 81.43% accuracy. <br>
# 
# Here are some short recommendations for your hypothetical employer:
# 
# Monitoring and Maintenance <br>
# Security and Privacy <br>
# User Interface and Accessibility <br>
# Documentation and Knowledge Transfer <br>
# Model Evaluation and Fine-Tuning <br>
# Data Quality and Preprocessing <br>
# Interpretability and Explainability <br>
# 
# 
# Concerns about the data or classification model can include:
# 
# Data Bias <br>
# Overfitting or Underfitting <br>
# Model Robustness <br>
# Ethical Considerations <br>
# 

# # My Thoughts on assumptions

# Here I am only mentioning the points with which I disagree and my reason for disagreeing.
# 
# - **Booking Lead Time and Cancellation**:
# While early bookings might have a higher chance of cancellation, it's crucial to explore the reasons behind early cancellations. It could be related to changes in plans or uncertainties rather than a direct correlation with the lead time.
# 
# - **Booking Duration and Cancellation**:
# While longer booking durations might indicate a commitment to the reservation, other factors like trip flexibility or the purpose of the stay could also play a role in cancellations.
# 
# - **Presence of Children and Cancellation**:
# While having children might be a factor, it's important to consider the nature of the booking (business, leisure) and whether the hotel is family-friendly. Other factors influencing cancellations should be explored.
# 
# - **Number of Booking Changes and Cancellation**:
# While more changes might indicate a higher commitment, it's essential to consider the nature of changes and whether they reflect customer dissatisfaction or uncertainty.
# 
# - **Refundable Bookings and Cancellation**:
# Refundable bookings might indeed have a higher cancellation rate, but this could also be influenced by factors like trip flexibility or changing circumstances.

# In[ ]:




