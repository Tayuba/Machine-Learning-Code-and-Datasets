import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

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

# Using SimpleImputer to find the meadian
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