import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Glance through titanic data
df = pd.read_csv("titanic.csv")
print(df)

# From the data, some variables have no impact on the survival rate, therefore i will drop them
df.drop(["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"], axis=1, inplace=True)
print(df)

# Split the data set into target and input
target = df.Survived
input = df.drop("Survived", axis=1)

# Now it can be seen that Sex column is a string, therefore i have to convert it into dummy variables
dummies = pd.get_dummies(input.Sex)
print(dummies)

# I will concatenate dummies to input
input = pd.concat([input, dummies], axis=1)
print(input)

# Finally i will drop the sex column
input.drop("Sex", axis=1, inplace=True)
print(input)

# Check and see if there are NaNa in the data
NaN_check = input.columns[input.isna().any()]
print(NaN_check)
"""The age Column has NaNa, therefore I have to handle this NaNa. What I am going to do is to fill the mean of age at
NaNa"""
# Handling NaNa
input.Age = input.fillna(input.Age.mean())
print(input)

"""I will split my data into train and test with sklearn with 20% test and 80% train"""
x_train, x_test, y_train, y_test = train_test_split(input, target, test_size=0.2)
print(len(x_train))

"""I will create Naive base class using Gaussian Naive Base"""
model = GaussianNB()

# I will have to train my model now
model.fit(x_train, y_train)

# I will check my score
model_score = model.score(x_test, y_test)
print(model_score)

"""Print my y test """
print(y_test[:10], "\n")

# I will predict first 10
predict_model = model.predict(x_test[:10])
print(predict_model)

# Check probability of survival
prob_survive = model.predict_proba(x_test[:10])
print(prob_survive)
