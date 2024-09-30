#!/usr/bin/env python
# coding: utf-8

# # 1. Problem Definition
# 
# - The problem says that we have to build best the best price prediction model we can given the data at hand.

# * Data representing over 11000 entries.
# * The obvious goal is to predict which customers are likely to churn, given a short set of attributes for each customer.

# # 2. Data Collection
# Gather the necessary data to solve the problem

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[13]:


df= pd.read_csv("/Users/shobhitdhanyakumardiggikar/Downloads/car_price_data (2).csv")


# In[3]:


df.info()


# In[14]:


df.head() 


# # 3. Data Processing
# Handle missing data, remove outliers and null values, transform data formats, perform feature engineering, etc.

# In[6]:


df.isnull().sum()


# In[18]:


# Remove rows with null values (missing data).
df = df.dropna()

# Optionally, you can reset the index if you want to reindex the DataFrame.
df.reset_index(drop=True, inplace=True)

# Save the cleaned DataFrame back to a CSV file.
cleaned_csv_file_path = 'cleaned_dataset.csv'
df.to_csv(cleaned_csv_file_path, index=False)

print(f'Null values removed. Cleaned data saved to {cleaned_csv_file_path}.')


# In[19]:


df.isnull().sum()


# # 4. Checking the duplicate values in data

# In[20]:


# Check for duplicates and get the duplicated rows
duplicates = df[df.duplicated(keep=False)]

# Print or further process the duplicate rows as needed
print(f"There are {len(duplicates)} rows of duplicated data")
print("---")
print(duplicates)


# In[27]:


df = df.drop_duplicates()


# In[28]:


df.info()


# # 4. Outliers detection and their treatment

# In[29]:


from scipy import stats
target = df["MSRP"]
# Calculate the IQR (Interquartile Range)
Q1, Q3 = stats.mstats.mquantiles(target, prob=[0.25, 0.75])
IQR = Q3 - Q1

# Define a lower bound and an upper bound for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the DataFrame to keep only non-outlier rows
df = df[(target >= lower_bound) & (target <= upper_bound)]


# In[30]:


df.info()


# # Train and Test Split 

# In[54]:


from sklearn.model_selection import train_test_split

# Define your predictors (X) and target (y)
# Example: Assuming 'Make', 'Year', 'Engine HP', 'Engine Cylinders' are your predictors
X = df[['Make', 'Year', 'Engine HP', 'Engine Cylinders']]
y = df['MSRP']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now you can move on to step 2: Selecting Predictors


# In[55]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Data Preparation
X1 = df[['Engine Cylinders']]  # Explanatory variable (Engine Cylinders)
y1 = df['MSRP']  # Target variable

# Split the data into training and testing sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Step 2: Train the Model
model1 = LinearRegression()

# Train the model
model1.fit(X1_train, y1_train)

# Step 3: Evaluate the Model

# Predict using the model
y1_pred = model1.predict(X1_test)

# Calculate SSE (Sum of Squared Errors)
sse1 = mean_squared_error(y1_test, y1_pred) * len(y1_test)

# Parameter estimate for 'Engine Cylinders'
parameter_estimate1 = model1.coef_[0]

# Report the results
print(f'Sum of Squared Errors (SSE) for Model 1: {sse1}')
print(f'Parameter estimate for Engine Cylinders: {parameter_estimate1}')


# In[ ]:





# In[ ]:





# # 5. Exploratory Data Analysis
# Explore the distribution of data, relationships between variables, key features, etc., using visualization and statistical methods.

# In[44]:


import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset into a Pandas DataFrame
df = pd.read_csv('cleaned_dataset.csv')

# Plot 1: Bar chart of Engine Fuel Type
engine_fuel_counts = df['Engine Fuel Type'].value_counts()[:10]
engine_fuel_counts.plot(kind='bar', color='Green')
plt.title('Top 10 Engine Fuel Types')
plt.xlabel('Engine Fuel Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Plot 2: Scatterplot of Engine HP vs. Highway MPG
plt.figure(figsize=(10, 8))
plt.scatter(df['Engine HP'], df['highway MPG'], color='Blue', alpha=0.5)
plt.title('Scatterplot of Engine HP vs. Highway MPG')
plt.xlabel('Engine HP')
plt.ylabel('Highway MPG')
plt.show()

# Plot 3: Pie Chart of Market Categories
market_category_counts = df['Market Category'].value_counts()[:5]
market_category_counts.plot(kind='pie', autopct='%1.1f%%', colors=['green', 'blue', 'lightcoral', 'Yellow', 'lightpink'])
plt.title('Top 5 Market Categories')
plt.show()


# Model 1 - Engine HP:
# In this section, we construct a basic linear regression model with "Engine HP" as the independent variable. The aim is to comprehend the influence of "Engine HP" on the dependent variable, MSRP (Manufacturer's Suggested Retail Price). We assess the model's performance using the Sum of Squared Errors (SSE), which measures how well the model fits the data. A lower SSE signifies a more accurate fit of the model to the data.
# 
# The parameter estimate for "Engine HP" indicates how a one-unit change in Engine HP affects MSRP. This estimate helps us grasp the strength and direction of the relationship between Engine HP and MSRP.
# 
# Model 2 - Highway MPG:
# In this section, we create another straightforward linear regression model, this time using "highway MPG" as the independent variable. The objective is to investigate the impact of "highway MPG" on the dependent variable, MSRP. Similar to Model 1, we use SSE to evaluate Model 2. The parameter estimate for "highway MPG" illustrates how MSRP changes with a one-unit increase in highway MPG.
# 
# These two models offer insights into the connections between the chosen features (Engine HP and highway MPG) and the target variable, MSRP. The assessment of SSE helps us gauge the model's accuracy, and the parameter estimates reveal the nature and magnitude of these relationships. This information is valuable for making predictions based on individual features.

# # Machine Learning Models

# In[47]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load your dataset into a Pandas DataFrame
df = pd.read_csv('cleaned_dataset.csv')

# Define the features and target variable
features = ['Engine HP', 'highway MPG']
target = 'MSRP'

# Loop through the features and create linear regression models
for feature in features:
    # Extract the feature as the explanatory variable
    feature_data = df[[feature]]
    
    # Create a Linear Regression model and fit it
    model = LinearRegression()
    model.fit(feature_data, df[target])
    
    # Make predictions using the model
    predictions = model.predict(feature_data)
    
    # Calculate the Sum of Squared Errors (SSE)
    sse = np.sum((df[target] - predictions) ** 2)
    
    # Get the parameter estimate (coefficient) for the feature
    parameter_estimate = model.coef_[0]
    
    # Print the results for each model
    print(f"Model for {feature}:")
    print(f"SSE: {sse:.2f}")
    print(f"Parameter Estimate ({feature}): {parameter_estimate:.2f}")
    print()


# In[49]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Initialize lists to store selected variables and SSEs
selected_vars_forward = []
sse_forward = []

# Define all available features
all_vars = ['Engine HP', 'Engine Cylinders', 'highway MPG', 'city mpg', 'Popularity']

# Create a copy of the dataset
data_copy = df.copy()

# Set the target variable
target = data_copy['MSRP']

# Step 1: Forward Variable Selection
while len(selected_vars_forward) < len(all_vars):
    min_sse = float('inf')
    best_var = None
    for var in all_vars:
        if var not in selected_vars_forward:
            # Fit a linear model with the selected variables
            temp_model = LinearRegression()
            temp_model.fit(data_copy[selected_vars_forward + [var]], target)

            # Make predictions
            predictions = temp_model.predict(data_copy[selected_vars_forward + [var]])

            # Calculate SSE
            sse = np.sum((target - predictions) ** 2)

            # If this SSE is better, update the best variable
            if sse < min_sse:
                min_sse = sse
                best_var = var

    # Add the best variable to the selected variables list
    selected_vars_forward.append(best_var)
    sse_forward.append(min_sse)

# Create a DataFrame to display the results
selection_results = pd.DataFrame({
    'Forward_Variables': selected_vars_forward,
    'SSE_Forward': sse_forward,
})

# Display the results
print(selection_results)


# In[50]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your dataset into a Pandas DataFrame
data = pd.read_csv('cleaned_dataset.csv')

# Step 2: Feature Selection/Engineering
# Select the predictors you believe are the best for your model
selected_predictors = data[['Engine HP', 'Engine Cylinders', 'highway MPG', 'city mpg', 'Popularity']]

# Step 3: Split Data
X = selected_predictors
y = data['MSRP']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build the Model
model = LinearRegression()

# Step 5: Train the Model
model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Calculate SSE
sse = mean_squared_error(y_test, y_pred) * len(y_test)

# Report the SSE (Sum of Squared Errors)
print(f"SSE: {sse:.2f}")


# # Predictor Selection for Regression Models
# 
# Selecting the appropriate predictors for a regression model is a critical step that significantly affects the model's performance and comprehensibility. Several key factors play a role in predictor selection:
# 
# 1. Relevance to the Problem: Opt for predictors that logically and reasonably impact the target variable. For instance, when predicting car prices, pertinent predictors could encompass factors like horsepower, fuel efficiency, and other car characteristics.
# 
# 2. Statistical Significance: Ensure that the chosen predictors exhibit a statistically significant connection with the target variable, typically indicated by low p-values in linear regression.
# 
# 3. Domain Expertise: Harness domain expertise to make informed decisions regarding which predictors to include, steering clear of irrelevant or redundant variables.
# 
# 4. Feature Importance: Some machine learning models offer feature importance scores, aiding in the identification of predictors with the most substantial influence.
# 
# 5. Data Exploration and Visualization: Techniques like exploratory data analysis and visualization unveil relationships between predictors and the target, offering valuable insights.
# 
# 6. Model Performance: Experiment with different predictor subsets to find the combination that yields the most favorable model performance. Techniques like forward selection or backward elimination can be helpful.
# 
# 7. Guard Against Overfitting: Be cautious about including too many predictors, as it can lead to overfitting. Select relevant predictors to mitigate this issue.
# 
# 8. Emphasize Interpretability: Prioritize simplicity and interpretability, as models with fewer predictors tend to be more understandable.
# 
# 9. Consider Data Availability and Quality: Practical considerations like data availability and data quality should influence predictor selection.
# 
# In summary, predictor selection is a delicate balance between model effectiveness and comprehensibility. The aim is to construct a regression model that accurately predicts the target variable while remaining practical and applicable in real-world scenarios.

# # Conclusion

# - Are you satisfied with your final model’s performance?  Why or why not?
# 
# Although the model forms the basis for price predictions of vehicles, I must acknowledge that I find its performance somewhat lacking. The relatively high Sum of Squared Errors (SSE) suggests a significant level of variability between the predicted and actual prices. This implies that the model might not encompass all the pertinent factors that impact vehicle pricing.
# 
# 

# - What would you do differently if you could?
# 
# As I continue with the coursework, I intend to explore more intricate models that have the ability to capture non-linear connections. This may include experimenting with ensemble models such as Random Forest or Gradient Boosting, along with neural networks. Additionally, I'll dedicate more attention to data cleansing, managing outliers, and resolving missing values.

# - Next Steps:
# Feature Engineering and Selection: We may need to delve deeper into feature engineering and potential feature interactions. Reassessing the significance of each feature in the model is also essential.
# 
# Model Selection and Hyperparameter Tuning: It's worth considering the exploration of more advanced models or refining existing ones to improve predictive capability.
# 
# Data Augmentation: Expanding the dataset to include a wider variety of vehicles and market situations has the potential to boost the model's accuracy.
# 
# Integration of External Data: Incorporating external data sources, such as economic indicators or consumer sentiment, could offer valuable context for pricing dynamics.

# - Based on my findings, I would want to communicate to business leadership in the automotive industry the following key points:
# 
# Model Capabilities: I would emphasize the model's ability to predict car prices effectively, utilizing key attributes such as Engine HP, Engine Cylinders, and fuel efficiency. It's crucial to underscore that the model serves as a robust initial step in estimating prices.
# 
# Opportunities for Enhancement: It's essential to convey that although the model performs reasonably well, there exist prospects for improvement. This involves exploring advanced modeling techniques and fine-tuning the feature selection to enhance predictive precision.
# 
# Interpretability and Alignment: I would underline the significance of the model being interpretable and in alignment with the pricing strategies of the business. The model should not only offer accurate predictions but also provide insights into the reasoning behind specific price estimations.
# 
# Continuous Improvement: I would recommend a continuous process of model development, assessment, and refinement. The industry should remain open to ongoing enhancements in predictive accuracy and model pertinence.
