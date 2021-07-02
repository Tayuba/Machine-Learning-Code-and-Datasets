import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


car_prices = {
                "Mileage": [69000, 35000, 57000, 22500, 46000, 59000, 52000, 72000, 91000, 67000, 83000, 79000,
                            59000, 58780, 82450, 25400, 28000, 69000, 87600, 52000],
                "Age(yrs)": [6, 3, 5, 2, 4, 5, 5, 6, 8, 6, 7, 7, 5, 4, 7, 3, 2, 5, 8, 5],
                "Sell Price($)": [18000, 34000, 26100, 40000, 31500, 26750, 32000, 19300, 12000, 22000, 18700,
                            19500, 26000, 27500, 19400, 35000, 35500, 19700, 12800, 28200]
}

# Converting Dictionary to DataFrame
car_df = pd.read_csv("car_prices.csv")
df = pd.DataFrame(car_prices)
# print(df)

# DataFrame to CSV
df_csv = df.to_csv(index=False)
# print(df_csv)

# Create CSV File
try:
    with open("car_prices.csv", "w+") as file:
        file.write(df_csv)
except PermissionError:
    print("File exist\n")

# Read CSV File into DataFrame
df = pd.read_csv("car_prices.csv")

# Reading CSV file
print(car_df, "\n")


# Plot scatter graph to understand relationship between dependent and independent variable
plt.scatter(car_df.Mileage, car_df["Sell Price($)"])
# plt.show()
plt.scatter(car_df["Age(yrs)"], car_df["Sell Price($)"])
# plt.show()

# Getting x and y variables
x = car_df[["Mileage", "Age(yrs)"]]
y = car_df["Sell Price($)"]
print(x, "\n")
print(y, "\n")

# Splitting data into train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x_train, "\n")

# Creating linear regression model object
model = LinearRegression()

# Training the model
model.fit(x_train, y_train)

# Prediction
predict_model = model.predict(x_test)
print(predict_model)

# Checking accuracy of the model
accuracy_model = model.score(x_test, y_test)
print(accuracy_model)