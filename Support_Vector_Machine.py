import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Creating Iris data set
iris = load_iris()
print(dir(iris))
"""['DESCR', 'data', 'feature_names', 'filename', 'frame', 'target', 'target_names']"""

#Now let me view what feature_names contains
print(iris.feature_names)
"""['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']"""


# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df, "\n")

# I will now append my target to the dataframe
df["target"] = iris.target
print(df)

# Since the targets is in 0, 1, 2, let confirm the corresponding names of such numbers
print(iris.target_names)
"""['setosa' 'versicolor' 'virginica']"""

# Now append the names of the target to the dataframe
df["flower_names"] = df.target.apply(lambda i: iris.target_names[i])
print(df)

"""Exporting this data into CSV file"""
# DataFrame to CSV
df_csv = df.to_csv(index=False)
print(df_csv)

# Create CSV File
try:
    with open("Iris_data.csv", "w+") as file:
        file.write(df_csv)
except PermissionError:
    print("File exist\n")

# Read CSV File into DataFrame
df = pd.read_csv("Iris_data.csv")
print(df, "\n")

# I will now visualize the dat on scatter diagram, i will divide the the set into three(3)
df0 = df[df.target == 0]
df1 = df[df.target == 1]
df2 = df[df.target == 2]

plt.scatter(df0["sepal length (cm)"], df0["sepal width (cm)"], color="red", marker="+")
plt.scatter(df1["sepal length (cm)"], df1["sepal width (cm)"], color="blue", marker="+")
# plt.scatter(df2["sepal length (cm)"], df2["sepal width (cm)"], color="green", marker="+")
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.show()

plt.scatter(df0["petal length (cm)"], df0["petal width (cm)"], color="red", marker="+")
plt.scatter(df1["petal length (cm)"], df1["petal width (cm)"], color="blue", marker="+")
# plt.scatter(df2["sepal length (cm)"], df2["sepal width (cm)"], color="green", marker="+")
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.show()

# Time to train the model, first split the data set
x = df.drop(df[["target", "flower_names"]], axis=1)
print(x)
y = df.target
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = SVC()
model.fit(x_train, y_train)

# Check accuracy of the model
accuracy_model = model.score(x_test,y_test)
print(accuracy_model)

