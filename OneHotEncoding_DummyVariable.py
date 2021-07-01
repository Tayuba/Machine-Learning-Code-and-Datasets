import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import linear_model

homeprice = {
                "town": ["Greater Accra", "Greater Accra", "Greater Accra", "Greater Accra", "Greater Accra",
                         "Ashanti Region", "Ashanti Region", "Ashanti Region", "Ashanti Region", "Upper East",
                         "Upper East", "Upper East", "Upper East"],
                "area": [2600, 3000, 3200, 3600, 4000, 2600, 2800, 3300, 3600, 2600, 2900, 3100, 3600],
                "price": [550000, 565000, 610000, 680000, 725000, 585000, 615000,
                          650000, 710000, 575000, 600000, 620000, 695000]
}

# Converting Dictionary to DataFrame
df = pd.DataFrame(homeprice)
# print(df)

# DataFrame to CSV
df_csv = df.to_csv(index=False)
# print(df_csv)

# Create CSV File
try:
    with open("home_prices.csv", "w+") as file:
        file.write(df_csv)
except PermissionError:
    print("File exist\n")

# Read CSV File into DataFrame
df = pd.read_csv("home_prices.csv")

"""------------------------------------------Working with Dummy Variable--------------------------------------------"""
# Using get dummy function
dummies_df = pd.get_dummies(df.town)

# Concatenate dummies to original data frame
merged_df = pd.concat([df, dummies_df], axis=1)
print(merged_df, "\n")

# Drop towm and Ashanti Region column (one of the dummy variables)
df_without_town = merged_df.drop(["town", "Ashanti Region"], axis=1)
print(df_without_town, "\n ")

# Creating object of model
model = linear_model.LinearRegression()

"""Now I want to input my "x" variable for training, this is nothing but all the colums in my df_without_town
 except price(y). First I have to drop the price colum"""
# Drop price
x = df_without_town.drop(["price"], axis=1)
print(x, "\n")

# y becomes my price
y = df_without_town.price
print(y, "\n")

# Train my model
model.fit(x,y)

# Prediction
predicted_value = model.predict([[3400, 0, 0]])
print(predicted_value,"\n")

# Checking model accuracy
accurate_model = model.score(x, y)
print(accurate_model,"\n")


"""---------------------------------------------Working One Hot Encoding--------------------------------------------"""
df_label = df
print(df, "\n")

# Doing label encoding on the town column
label_encoder = LabelEncoder()
df_label.town = label_encoder.fit_transform(df.town)
print(df_label, "\n")

# Get x variable
x = df_label[["town", "area"]].values
print(x,"\n")

# Get y variable
y = df_label[["price"]]
print(y,"\n")

# Create One Hot Encoder object and Dummy
Dummy_OHE = ColumnTransformer(
    [('OHE', OneHotEncoder(categories='auto'), [0])],
    remainder='passthrough'
)

dx = Dummy_OHE.fit_transform(x)
# dropping one column of the dummy array
new_x = dx[:,1:]
print(new_x, "\n")

# Train the model
model.fit(new_x, y)

# Prediction
predicted_model = model.predict([[0, 0, 3400]])
print(predicted_model)