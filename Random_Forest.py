import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sn



# Load the digits data
digits = load_digits()

# Visualize the data
plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])
# plt.show()

# Create DataFrame from the data set
df = pd.DataFrame(digits.data)
print(df)

# Append target to the DataFrame
df["target"] = digits.target
# print(df)

# Split the data set
x = df.drop(["target"], axis=1)
y = digits.target
# print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# print(len(x_test))

# Create random forest model
model = RandomForestClassifier()
# Train the model
model.fit(x_train, y_train)

# Checking performance
accuracy_model = model.score(x_test, y_test)
print(accuracy_model)

# Plot true and predicted value, to check where model fall short
y_predict = model.predict(x_test)

# Create a confusing metrics to compare the performance of my model
conf_matrix = confusion_matrix(y_test, y_predict)
print(conf_matrix)

# Plot the comparison
plt.figure(figsize=(10, 10))
sn.heatmap(conf_matrix, annot=True)
plt.xlabel("Predicted Value")
plt.ylabel("TRUE Value")
plt.show()