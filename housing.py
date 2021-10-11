import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn import svm
from sklearn.model_selection import GridSearchCV


""" Source of data"""
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


""" Function to fetching Housing Data"""
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


fetch_housing_data()

""" Function to Load and concert the data file to pandas DataFrame"""
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


"""Glance through the Housing Data"""

housing = load_housing_data()
# View data
housing.hist(bins=50, figsize=(20, 15))
# plt.show()

# Check the first 5 of the dataset
first_5_data = housing.head()
print(first_5_data,"\n")

# Check insight of the information on the dataset
info_of_data = housing.info()
print(info_of_data,"\n")

# A deep view into ocean_proximity
category_in_data = housing["ocean_proximity"].value_counts()
print(category_in_data,"\n")

# Check the statistics on the dataset
summary_numof_data = housing.describe()
print(summary_numof_data,"\n")


"""Understanding and knowing the best method of splitting of datasets into train and test"""
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0.,1.5,3.0,4.5,6., np.inf], labels=[1,2,3,4,5])
housing["income_cat"].hist()
# plt.show()

# Random split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# Stratified split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    strat = strat_test_set["income_cat"].value_counts()/ len(strat_test_set)


"""Comparing bias in randomly split with stratified split with original income column"""
# Original column of average income cat
print(housing["income_cat"].value_counts()/ len(housing))
# Random split average of income cat column
print(test_set["income_cat"].value_counts()/ len(test_set))
# stratified split average of income cat column
print(strat)

"""Drop income_cat in both strat_train_set and strat_test_set, since it was only used for analysis"""
for drp in (strat_train_set, strat_test_set):
    drp.drop("income_cat", axis=1, inplace=True)

"""Create a copy of strat_train_set for manipulation purpose"""
strat_train_set = strat_train_set.copy()
print(strat_train_set)

"""Visualizing Geographical Data"""
strat_train_set.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# plt.show()

"""Now compare population and visualizing Geographical with the median house value"""
strat_train_set.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1, s=strat_train_set["population"]/100,
                     label="population", figsize=(10, 7), c="median_house_value", cmap=plt.get_cmap("jet"),
                     colorbar=True)
plt.legend()
# plt.show()

"""Since the dataset is not that large, corr() can be compute to check the correlation between very attributes pairs"""
corr_matrix = strat_train_set.corr()
# correlation against median house value
corr_MHV = corr_matrix["median_house_value"].sort_values(ascending=False)
print(corr_MHV)

"""Using Pandas to plot the correlations"""
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(strat_train_set[attributes], figsize=(12, 8))


"""The most promising attribute to predict median house is median income,let zoom median income using scatter plot"""
strat_train_set.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
# plt.show()

"""Attributes combinations"""
strat_train_set["rooms_per_household"] = strat_train_set["total_rooms"]/strat_train_set["households"]
strat_train_set["bedrooms_per_room"] = strat_train_set["total_bedrooms"]/strat_train_set["total_rooms"]
strat_train_set["population_per_household"]=strat_train_set["population"]/strat_train_set["households"]
# Now check the correlations between this new attributes with median house  value
corr_matrix = strat_train_set.corr()
corr_MHV = corr_matrix["median_house_value"].sort_values(ascending=False)
print(corr_MHV)


"""Divide the dataset into predictors(x) and labels(y)"""
# Predictor
x = strat_train_set.drop("median_house_value", axis=1)
print(x.isna().sum())
print()
# Labels
y = strat_train_set["median_house_value"].copy()
print(y.isna().sum())

"""Its clear that my predictor have some empty values in total bedrooms, i can either do three of these, 1.drop those
rows, 2.drop the entire attribute or column or 3. find the median and fill it in the empty values"""
# I will do the 3 by finding the median and filling it in the empty values row, to do this i have to drop the non
# numerical attributes from the train dataset
x_num = x.drop("ocean_proximity", axis=1)
print(x_num)

# Using SimpleImputer to find the median
imputer = SimpleImputer(strategy="median")
imputer.fit(x_num)


# Replacing missing values
x_replace = imputer.transform(x_num)
print(len(x_replace))

# Put transform values into dataframe
x_dataframe = pd.DataFrame(x_replace, columns=x_num.columns)
print(x_dataframe)


"""Handling Text and Categorical Attributes using OneHotEncoder"""
x_cat = strat_train_set[["ocean_proximity"]]

OHT = OneHotEncoder()
x_cat_OHT = OHT.fit_transform(x_cat).toarray()
print(x_cat_OHT)

"""Custom transformer"""

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(strat_train_set.values)
print(housing_extra_attribs)


"""Creating a pipeline for data transformation"""
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', MinMaxScaler()),
    ])

x_num_trans = num_pipeline.fit_transform(x_num)
print()

"""Handling both categorical and numerical columns at once"""
num_attribs = list(x_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
    ])
housing_prepared = full_pipeline.fit_transform(x)
print(housing_prepared)

"""Training and Evaluating Training Set"""
l_reg = LinearRegression()
l_reg.fit(housing_prepared, y)

"""Try some data on the model"""
some_data = x.iloc[:5]
some_labels = y.iloc[:5]
some_pre = full_pipeline.transform(some_data)

print("Predictions", l_reg.predict(some_pre))
print("Labels", list[some_labels])

d_reg = DecisionTreeRegressor()
d_reg.fit(housing_prepared, y)

print("Predictions", d_reg.predict(some_pre))
print("Labels", list[some_labels])

rand_reg = RandomForestRegressor()
rand_reg.fit(housing_prepared, y)

print("Predictions", rand_reg.predict(some_pre))
print("Labels", list[some_labels])

"""Save model"""
joblib.dump(rand_reg, "my_model.joblib")

"""Fine Tuning and best model selection"""

model_param = {
    # "svm": {
    #     "model": svm.SVC(gamma="auto"),
    #     "params": {
    #         "C": [1, 10, 20],
    #         "kernel": ["rbf", "linear"]
    #     }
    # },

    "random_forest": {
        "model": RandomForestRegressor(),
        "params": [
            {
            "n_estimators": [1, 5, 45],
            "max_features": [2, 4, 6, 16]
            },
            {
            "bootstrap": [False],
            "n_estimators": [3, 10],
            "max_features": [2, 3, 4]
            }
            ]

        },
    # "linear_regression": {
    #     "model": LinearRegression(),
    #     "params": {
    #         "C": [1, 5, 10],
    #
    #     }
    #
    # }
}


#Iterate through json object, use the GridSearchCV to select the best score, model and parameterd
scores = []
#
for model_name, model_parameter in model_param.items():
    clf = GridSearchCV(model_parameter["model"], model_parameter["params"], cv=5, return_train_score=False)
    clf.fit(housing_prepared, y)

    scores.append({
        "model": model_name,
        "best_score": clf.best_score_,
        "best_params": clf.best_params_
    })


# Convert the scores list into pandas DataFrame
df = pd.DataFrame(scores)
print(df)
print()

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
x_num = X_test.drop("ocean_proximity", axis=1)

X_test_prepared = full_pipeline.transform(X_test)

"""Evaluate the model on the best dataset"""
final_model = clf.best_estimator_



final_predictions = final_model.predict(X_test_prepared)
print(final_predictions)
print()

# Check the score
model_score = final_model.score(final_predictions, y_test)
print(model_score)

