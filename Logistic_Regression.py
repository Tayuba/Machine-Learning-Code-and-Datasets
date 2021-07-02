import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


Insurance_age ={
    "Age": [22, 25, 47, 52, 46, 56, 55, 60, 62, 61, 18, 28, 27, 29, 49, 55, 25, 58, 19, 18, 21, 26, 40, 45, 50, 54, 23],
    "Bought_Insurance": [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0]
}

# Converting Dictionary to DataFrame
Insurance_df = pd.DataFrame(Insurance_age)
print(Insurance_df)

# DataFrame to CSV
df_csv = Insurance_df.to_csv(index=False)
print(df_csv)

# Create CSV File
try:
    with open("Insurance.csv", "w+") as file:
        file.write(df_csv)
except PermissionError:
    print("File exist\n")

# Read CSV File into DataFrame
df = pd.read_csv("Insurance.csv")

# Read CSV File
Insurance_df = pd.read_csv("Insurance.csv")
print(Insurance_df)

# Plot the data for visualization
plt.scatter(Insurance_df.Age, Insurance_df.Bought_Insurance, marker="+", color="green")
plt.show()

# splitting data into train and test set
x_train, x_test, y_train, y_test = train_test_split(Insurance_df[["Age"]], Insurance_df.Bought_Insurance, test_size=0.1)
print(x_test)
# create object of the model
model = LogisticRegression()

# Train the model
model.fit(x_train, y_train)

# Try Making Predictions
Predict_model = model.predict(x_test)
print(Predict_model)

# Checking the accuracy of the model
accuracy_model = model.score(x_test, y_test)
print(accuracy_model)

# Check probability
pro_model = model.predict_proba(x_test)
print(pro_model)