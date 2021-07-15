import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sn
import warnings


# Load housing data
data_set = pd.read_csv("Melbourne_housing_FULL.csv")
print(data_set)

# filter the dataset
data_set = data_set[["Suburb", "Rooms", "Type", "Method", "SellerG", "Regionname", "Propertycount", "Distance",
                     "CouncilArea", "Bedroom2", "Bathroom", "Car", "Landsize", "BuildingArea", "Price"]]
print(data_set)

# Checking for NaN
NaN_Value = data_set.isna().sum()
print(NaN_Value, "\n")

# It is reasonable to fill the columns below with NaN with 0 since they are unique and we can assume and number for them
fill_zero = ["Propertycount", "Distance", "Bedroom2", "Bathroom", "Car"]
data_set[fill_zero] = data_set[fill_zero].fillna(0)
# Checking for NaN
NaN_Value = data_set.isna().sum()
print(NaN_Value)

# I will find the mean values and fill it with the NaN below
data_set["Landsize"] = data_set["Landsize"].fillna(data_set.Landsize.mean())
data_set["BuildingArea"] = data_set["BuildingArea"].fillna(data_set.Landsize.mean())
# Checking for NaN
NaN_Value = data_set.isna().sum()
print(NaN_Value)

# Finally i will drop the rest of NaN values
data_set.dropna(inplace=True)
# Checking for NaN
NaN_Value = data_set.isna().sum()
print(NaN_Value)
print(data_set)

"""Now my data is clean, I will now convert all text to number in my dataset using get dummies from pandas """
data_set = pd.get_dummies(data_set, drop_first=True)
print(data_set)

# Divide my data set into x and y
x_data = data_set.drop("Price", axis=1)
y_data = data_set.Price

# Train my data
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=2)

# Linear Regression Model
Reg = LinearRegression()
Reg.fit(x_train, y_train)

# Check the score
Reg_score = Reg.score(x_test, y_test)
print(Reg_score)
Reg_score = Reg.score(x_train, y_train)
print(Reg_score)

"""From the score results it can be seen that my model is overfit, I will use Lasso Regression(L1) to improve the model"""
Lasso_Reg = linear_model.Lasso(alpha=50, max_iter=1000, tol=0.1)
Lasso_Reg.fit(x_train, y_train)

#Check score of lasso regression
Lasso_Reg_score = Lasso_Reg.score(x_test, y_test)
print(Lasso_Reg_score)
Lasso_Reg_score = Lasso_Reg.score(x_train, y_train)
print(Lasso_Reg_score)