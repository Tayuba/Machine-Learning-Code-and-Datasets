from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



age_income = {
    "Name":["Rob", "Micheal", "Mohan", "Ismail", "Kory", "Gautam", "David", "Andrea", "Brad", "Angelina", "Donald", "Tom"
        ,"Arnold", "Jared", "Stark", "Ranbir", "Dipika", "Priyanka", "Nick", "Alia", "Sid", "Abdul"],
    "Age": [27, 29, 29, 28, 42, 39, 41, 38, 36, 35, 37, 26, 27, 28, 29, 32, 40, 41, 43, 39, 41, 39],
    "Income": [70000, 90000, 61000, 60000, 150000, 155000, 160000, 162000, 156000, 130000, 137000, 45000, 48000, 51000,
        49500, 53000, 65000, 63000, 64000, 80000, 82000, 58000]
}

# Converting Dictionary to DataFrame
# car_df = pd.read_csv("age_income.csv")
df = pd.DataFrame(age_income)
# print(df)

# DataFrame to CSV
df_csv = df.to_csv(index=False)
# print(df_csv)

# Create CSV File
try:
    with open("age_income.csv", "w+") as file:
        file.write(df_csv)
except PermissionError:
    print("File exist\n")

# Read CSV File into DataFrame
df = pd.read_csv("age_income.csv")
print(df)

# Visualize my data
# plt.scatter(df.Age, df.Income)
# plt.show()

# Choosing K means
k_mean = KMeans(n_clusters=3)

# fit and predict the k mean to create y predict
y_predicted = k_mean.fit_predict(df[["Age", "Income"]])
print(y_predicted)

# append this y predicted to the data set
df["cluster"] = y_predicted
print(df)

# Using scatter graph to visualize my new data set, i will do that my first creating three(3) data set to plot
# df1 = df[df.cluster == 0]
# df2 = df[df.cluster == 1]
# df3 = df[df.cluster == 2]
#
#
# plt.scatter(df1.Age, df1["Income"], color="green", label="Income 0")
# plt.scatter(df2.Age, df2["Income"], color="red", label="Income 1")
# plt.scatter(df3.Age, df3["Income"], color="black", label="Income 2")
# plt.xlabel("Age")
# plt.ylabel("Income")
# plt.legend()
# plt.show()

# For better scale, I use MinMaxScaler, I have to scale my data parameters
scaler = MinMaxScaler()
scaler.fit(df[["Income"]])
df["Income"] = scaler.transform(df[["Income"]])

scaler.fit(df[["Age"]])
df.Age = scaler.transform(df[["Age"]])
print(df)

# Now I will use K mean to train again
k_mean = KMeans(n_clusters=3)
y_predicted = k_mean.fit_predict(df[["Age", "Income"]])
print(y_predicted)

# Append y predicted to the scaled data set
df["cluster"] = y_predicted
print(df)

# Now, I will plot this scaled data
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]


plt.scatter(df1.Age, df1["Income"], color="green", label="Income 0")
plt.scatter(df2.Age, df2["Income"], color="red", label="Income 1")
plt.scatter(df3.Age, df3["Income"], color="black", label="Income 2")
plt.scatter(k_mean.cluster_centers_[:,0], k_mean.cluster_centers_[:,1], color="blue", marker="*", label="centroid")
plt.xlabel("Age")
plt.ylabel("Income")
plt.legend()
plt.show()

# Using Eldow Technique
K_range = range(1,11)
# sume of square error
sse = []
for k in K_range:
    K_mean = KMeans(n_clusters=k)
    K_mean.fit(df[["Age", "Income"]])
    # inertia give sse
    sse.append(K_mean.inertia_)
print(sse)

# Plotting K to visualize it on graph
plt.xlabel("K")
plt.ylabel("Sum of Square Error")
plt.plot(K_range, sse)
plt.show()