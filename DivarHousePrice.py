#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns


# In[2]:


# Read the CSV file and store it in the variable divar_df
divar_df = pd.read_csv("file:///C:/Users/test/Downloads/DivarHousePrice.csv")


# In[3]:


# Convert necessary columns to numeric data type
for column in divar_df.select_dtypes(include=['object']).columns:
    divar_df[column] = pd.to_numeric(divar_df[column], errors='coerce')

# Create a LabelEncoder
label_encoder = LabelEncoder()

# Convert 'Address' column to numeric data type
divar_df['Address'] = label_encoder.fit_transform(divar_df['Address'])

print(divar_df.dtypes)


# In[4]:


# Fill missing values in 'Area' column with 0
divar_df['Area'].fillna(0, inplace=True)
# Count remaining missing values in the DataFrame
divar_df.isnull().sum()


# In[5]:


# Display the first few rows of the DataFrame
print(divar_df.head())

# Display information about the DataFrame
print(divar_df.info())

# Display descriptive statistics of the DataFrame
print(divar_df.describe())

# Display the shape of the DataFrame
print(divar_df.shape)


# In[6]:


# Get the column names of the DataFrame
column_names = divar_df.columns


# In[7]:


# Assign 'Price' column values back to themselves (no change)
divar_df['Price'] = divar_df['Price'].values


# In[8]:


# Display descriptive statistics for the 'Price' column
print(divar_df['Price'].describe())

# Display value counts for the 'Room' column
print(divar_df['Room'].value_counts())


# In[9]:


# Plot histogram of 'Price' column
print(divar_df['Price'].hist())


# In[10]:


# Plot scatter plot of 'Room' versus 'Price'
plt.scatter(divar_df['Room'], divar_df['Price'])
plt.xlabel('Room')
plt.ylabel('Price')
plt.show()


# In[11]:


x = divar_df[['Area', 'Room', 'Parking','Warehouse','Elevator','Address',]]
y = divar_df['Price']


# In[12]:


# Split the data into training and testing sets
# x: features, y: target variable
# test_size=0.2 means 20% of the data will be used for testing
# random_state=0 ensures reproducibility of the split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size =0.2,random_state = 0)


# In[13]:


# Print the shapes of the training and testing data sets
print("xtrain shape : ", xtrain.shape)
print("xtest shape : ", xtest.shape)
print("ytrain shape : ", ytrain.shape)
print("ytest shape : ", ytest.shape)


# In[14]:


# Create a Linear Regression model
# Fit the model using the training data
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)


# In[15]:


# predicting the test set results
y_pred = regressor.predict(xtest)


# In[16]:


# Plotting Scatter graph to show the prediction
# results - 'ytrue' value vs 'y_pred' value
plt.scatter(ytest, y_pred, c = 'green')
plt.xlabel("Price: in $1000's")
plt.ylabel("Predicted value")
plt.title("True value vs predicted value : Linear Regression")
plt.show()


# In[17]:


# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(ytest, y_pred)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(ytest,y_pred)

print("Mean Square Error : ", mse)
print("Mean Absolute Error : ", mae)


# In[18]:


# Set the size of the plot
plt.figure(figsize=(12, 6))

# Plot a line plot using seaborn
sns.lineplot(x='Room', y='Price', data=divar_df)

# Set title and labels for the plot
plt.title('divar')
plt.xlabel('Room')
plt.ylabel('Price')
plt.show()


# In[19]:


# Calculate correlation coefficients
correlation = divar_df['Price'].corr(divar_df['Area'])
print(correlation)
correlation = divar_df['Price'].corr(divar_df['Room'])
print(correlation)
correlation = divar_df['Price'].corr(divar_df['Parking'])
print(correlation)
correlation = divar_df['Price'].corr(divar_df['Warehouse'])
print(correlation)
correlation = divar_df['Price'].corr(divar_df['Elevator'])
print(correlation)
correlation = divar_df['Price'].corr(divar_df['Address'])
print(correlation)
correlation = divar_df['Price'].corr(divar_df['Price(USD)'])
print(correlation)


# In[20]:


#Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(divar_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap of Correlation Matrix')
plt.show()

