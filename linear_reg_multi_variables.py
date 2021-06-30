import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import linear_model

# Dictionary of house prices and features
home_price = { "Area": [2600, 3000, 3200, 3600, 4000],
               "bedrooms": [3, 4, np.nan, 3, 5],
                "Age": [20, 15, 18, 30, 8],
               "price": [550000, 565000, 610000, 595000, 760000],

}

# Converting Dictionary to DataFrame
df = pd.DataFrame(home_price)
# print(df)

# DataFrame to CSV
df_csv = df.to_csv(index=False)
# print(df_csv)

# Create CSV File
try:
    with open("home_prices_features.csv", "w+") as file:
        file.write(df_csv)
except PermissionError:
    print("File exist\n")

features_of_houses = pd.read_csv("home_prices_features.csv")
# print(features_of_houses)

"""Handling missing number of bedrooms for index 2, the best way will be to take the median.
After i can assume the median value will best fit my missing bedrooms for index 2"""

# Median in integer
bedrooms_median = math.floor(features_of_houses.bedrooms.median())
# print(bedrooms_median)

# Fill the missing value with the median value calculated above
features_of_houses.bedrooms = features_of_houses.bedrooms.fillna(bedrooms_median)
print(features_of_houses)

#  Creating linear regression class object
reg = linear_model.LinearRegression()

# Calling fit() to train my model
reg_houses = reg.fit(features_of_houses[["Area", "bedrooms", "Age"]], features_of_houses[["price"]])

print(reg_houses.coef_)
print(reg_houses.predict([[2500, 4, 5]]))