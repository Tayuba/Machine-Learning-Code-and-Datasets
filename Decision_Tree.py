import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree


comapny_salary = {
        "Company": ["google", "google", "google", "google", "google", "google", "abc pharma", "abc pharma", "abc pharma",
                    "abc pharma", "facebook", "facebook", "facebook", "facebook", "facebook", "facebook"],
        "Job": ["sale executive", "sale executive", "business manager", "business manager", "computer science",
                "computer science", "sale executive", "computer science", "business manager", "business manager",
                "sale executive", "sale executive", "business manager", "business manager", "computer science",
                "computer science"],
        "degree": ["bachelors", "masters", "bachelors", "masters", "bachelors", "masters", "masters", "bachelors",
                   "bachelors", "masters", "bachelors", "masters", "bachelors", "masters", "bachelors", "masters"],
        "Salary_more_than_100k": [0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
}
# print((comapny_salary))
# Converting Dictionary to DataFrame
df = pd.DataFrame.from_dict(comapny_salary)
print(df)

# DataFrame to CSV
df_csv = df.to_csv(index=False)
print(df_csv)

# Create CSV File
try:
    with open("company_salaries.csv", "w+") as file:
        file.write(df_csv)
except PermissionError:
    print("File exist\n")

# Read CSV File into DataFrame
df = pd.read_csv("company_salaries.csv")
print(df, "\n")

# Diving the data set into target and independent variable
inputs = df.drop("Salary_more_than_100k", axis=1)
print(inputs, "\n")
target = df.Salary_more_than_100k
print(target, "\n")

"""Now looking at the inputs, i have to convert my into into  numbers using One Hot Encoder. I am going to create
three(3) object of Encoder lable"""
label_company = LabelEncoder()
lable_job = LabelEncoder()
lable_degree = LabelEncoder()

# Creating one more column in my inputs
inputs["company_to_num"] = label_company.fit_transform(inputs["Company"])
inputs["job_to_num"] = label_company.fit_transform(inputs["Job"])
inputs["degree_to_num"] = label_company.fit_transform(inputs["degree"])
print(inputs)

# Drop all inputs that are not numbers, to do this i will create a new number inputs_num
inputs_nums = inputs.drop(["Company", "Job", "degree"], axis=1)
print(inputs_nums, "\n")

# Splitting the data set into train and test sets
x_train, x_test, y_train, y_test = train_test_split(inputs_nums, target, test_size=0.2)
print(x_train)
print(x_test, "\n")

# Now I will train my data set using tree module
model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)

# Check model score
score_model = model.score(x_test, y_test)
print(score_model)

# Predict my model
predicted_model = model.predict([[2, 0, 1]])
print(predicted_model)

