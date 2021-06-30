import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# Reading CSV File with Pandas
df = pd.read_csv("price.csv")
print(df)



# Predicting the Price and Training
reg = linear_model.LinearRegression().fit(df[["Area"]],df.Price)

# Plot Scatter Graph to Observe the data
plt.scatter(df.Area, df.Price, color="red", marker="+")
plt.plot(df.Area, reg.predict(df[["Area"]]), color="blue")
plt.xlabel("Area(sqr)")
plt.ylabel("Price(US$)")
plt.show()

# Now let assume you have a list of area of houses you want to predict prices the onto csv file at once
list_of_areas = {"Area": [1000, 1500, 2300, 3540, 4120, 4560, 5490, 3460, 4750, 2300, 9000, 8800, 7100]
                }
# Converting dictionary to Dataframe
area_to_predict_csv = pd.DataFrame(list_of_areas).to_csv(index=False)

# Creating a CSV file
with open("list_of_area.csv", "w+") as file:
    file.write(area_to_predict_csv)

# Reading the CSV file
area_lists = pd.read_csv("list_of_area.csv")
print(area_lists)

# Predict all the list of areas
all_price= reg.predict(area_lists)
# print(all_price)

# Store the all the prices in a new column
area_lists["Prices"] = all_price
print(area_lists)

# Export all the prices to my csv file
area_lists.to_csv("list_of_area.csv", index=False)