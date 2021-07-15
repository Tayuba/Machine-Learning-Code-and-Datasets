import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load the iris data set into DataFrame
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["flower"] = iris.target
df["flower"] = df["flower"].apply(lambda x : iris.target_names[x])
print(df)

"""the traditional method of solving this problem is by splitting the data set, train and then predict it"""
# Split the data set into train and test sets
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

# Assuming i am using SVM
model = svm.SVC(kernel="rbf", C=30, gamma="auto")
model.fit(x_train, y_train)

# Check the score
model_score = model.score(x_test, y_test)
print(model_score)

"""Because my score keep changing based on the train set at each time, i will use KFold method"""
"""using a for loop for finding the best score, this approach is not the best because the more the parameters 
the more my loop"""
kernels = ["rbf", "linear"]
C = [1, 10, 20]
avg_scores = {}
for kval in kernels:
    for cval in C:
        cv_scores = cross_val_score(svm.SVC(kernel=kval, C=cval, gamma="auto"), iris.data, iris.target, cv=5)
        avg_scores[kval + "_" + str(cval)] = np.average(cv_scores)
print(avg_scores)

"""sklearn learn has a function call GridSearchCV which will do the same thing as the code in the above"""
classifier = GridSearchCV(svm.SVC(gamma="auto"), {
    "C": [1, 10, 20],
    "kernel": ["rbf", "linear"]
}, cv=5, return_train_score=False)

classifier.fit(iris.data, iris.target)
crossVal_results = classifier.cv_results_
print(crossVal_results)

"""For better visualization, i will save my cross validation results in DataFrame"""
df = pd.DataFrame(crossVal_results)
print(df)

# I do not need these parameters in my calculations, so i will drop them
df = df[["param_C", "param_kernel", "mean_test_score"]]
print(df)

"""Now, that I am done with hyper parameter tuning, let see how to select best model for a given problem. I will be 
using these three classifiers SVM, RandomForestClassifier and LogisticRegression"""
# I will create json object which contain all the three classifiers as my model parameters
model_param = {
    "svm": {
        "model": svm.SVC(gamma="auto"),
        "params": {
            "C": [1, 10, 20],
            "kernel": ["rbf", "linear"]
        }
    },

    "random_forest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [1, 5, 10]

            }

        },
    "logistic_regression": {
        "model": LogisticRegression(solver="liblinear", multi_class="auto"),
        "params": {
            "C": [1, 5, 10],

        }

    }
}

#Iterate through json object, use the GridSearchCV to select the best score, model and parameterd
scores = []

for model_name, model_parameter in model_param.items():
    clf = GridSearchCV(model_parameter["model"], model_parameter["params"], cv=5, return_train_score=False)
    clf.fit(iris.data, iris.target)
    scores.append({
        "model": model_name,
        "best_score": clf.best_score_,
        "best_params": clf.best_params_
    })
# Convert the scores list into pandas DataFrame
df = pd.DataFrame(scores)
print(df)