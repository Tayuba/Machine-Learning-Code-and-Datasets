from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


# Load digits data sets
digits = load_digits()

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)

"""I will use different classifier to compare the performance of the various model"""
# Logistic Regression
Log_Re = LogisticRegression(solver="lbfgs", max_iter=4000)
Log_Re.fit(x_train, y_train)
Log_score = Log_Re.score(x_test, y_test)
print(Log_score)

# SVM
svm = SVC()
svm.fit(x_train, y_train)
svm_score = svm.score(x_test, y_test)
print(svm_score)

# Randome Forest
rand_forest = RandomForestClassifier()
rand_forest.fit(x_train, y_train)
rand_score = rand_forest.score(x_test, y_test)
print(rand_score)


"""It can be seen from the above that i can measure the performance of each model by calling it score anytime i want
to measure. Know I am going to use the better way to do it, with K_Fold"""
K_fold = KFold(n_splits=3)
stra_Kfold = StratifiedKFold(n_splits=3)


# Create Function to hold perform score calculations
def model_score(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

# Create array of all the model scores
Logi_score = []
svms_score = []
randF_score = []

for train_indix, test_index in K_fold.split(digits.data):
    x_train, x_test, y_train, y_test = digits.data[train_indix], digits.data[test_index], \
                                       digits.target[train_indix], digits.target[test_index]
    Logi_score.append(model_score(LogisticRegression(solver="lbfgs", max_iter=4000), x_train, x_test, y_train, y_test))
    svms_score.append(model_score(SVC(), x_train, x_test, y_train, y_test))
    randF_score.append(model_score(RandomForestClassifier(n_estimators=40), x_train, x_test, y_train, y_test))

print("",Logi_score,"\n",svms_score,"\n",randF_score, "\n")

"""The function above work perfect, but it a whole lot of code, there is a function call croos_val_score, this make 
the  code short and clear  """
LR = cross_val_score(LogisticRegression(solver="lbfgs", max_iter=4000), digits.data, digits.target)
SVMS = cross_val_score(SVC(), digits.data, digits.target)
RF = cross_val_score(RandomForestClassifier(n_estimators=40), digits.data, digits.target)
print("", LR, "\n", SVMS, "\n", RF)