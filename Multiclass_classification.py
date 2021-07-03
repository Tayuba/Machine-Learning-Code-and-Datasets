from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
import seaborn as sn



"""I am going to use in built hand written in sklearn for this exercise, this datasets is made up of 1797 8x8 images
of hand-written digit. In order for to utilize an 8x8 figure like this we would have to transform it into a feature
vector with length 64"""

# Creating object of load digits
digits = load_digits()
# let check what this digits contains
print(dir(digits))
"""['DESCR', 'data', 'feature_names', 'frame', 'images', 'target', 'target_names']"""

# let check the first elements in the data
first_element = digits.data[0]
print(first_element)
""" As seen it contains 64 elements, dimention of 8x8 size"""

# Now let check first 5 of this written images
for i in range(5):
    dig_images = plt.matshow(digits.images[i])
    print(dig_images)
#     plt.show()

# Finally let check the target of the first 5 images
image_targets = digits.target[:5]
print(image_targets)

# Splitting the data set into train and test sets
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)
print(len(x_train))
print(len(x_test))

# Using logistic regression to train the model
model = LogisticRegression(solver="lbfgs", max_iter=4000)
model.fit(x_train, y_train)

# Check the score of the model
model_score = model.score(x_test, y_test)
print(model_score)

# Now let predict using our x test
predict_model = model.predict(x_test)
print(predict_model)

# To find out where the model is not doing well
confusion_mat = confusion_matrix(y_test, predict_model)
print(confusion_mat)

# Let visualized the confusion matrix in seaborn
plt.figure(figsize=(10, 10))
sn.heatmap(confusion_mat, annot=True)
plt.xlabel("Predicted Value")
plt.ylabel("True Value")
plt.show()