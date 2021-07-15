import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


"""I want to explore the spam data first, as a rule i have to go through the data and see it make up"""
# Read my spam data
df = pd.read_csv("spam.csv")
print(df, "\n")

# I want to update myself with the number of spam and ham
cat_grp = df.groupby("Category").describe()
print(cat_grp)

"""I can see that my Category contains 'spam' and 'ham', as you may be aware machine learning works with numbers 
therefore, i will convert 'spam' and 'ham' into integer."""
# I will create 'spam column' for the category and use simple function to convert 'spam' and 'ham' into 0 and 1
df["spam"] = df["Category"].apply(lambda x: 1 if x=="spam" else 0)
print(df)

# I will divide my data into train and test set, with 25% test
x_train, x_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.25)
print(x_train)
print(y_train)


"""When i print my x_train it is seen that, all my messages are words as said ealier machine learning works with numbers
 therefore i will need to convert all this messages into integers based on the number of times they appear. The way i 
 will do this is by using 'Count Vectorized Technique'. This techniques gives me uniques words in each message, from 
 this i will build matrix with this unique words"""
# Creating maxtrix
vec = CountVectorizer()
x_train_convert = vec.fit_transform(x_train)
# View in maxtrix
print(x_train_convert.toarray())

# Using naive bayes classifier
model = MultinomialNB()
model.fit(x_train_convert, y_train)

"""I copy a spam message and non_spam message from my email to test the model, as seen below the first email is spam and
the second is not. my model predicted so which is true"""

emails = ["The Government Upto 20% discount - Solve Your Debt Problems "
         "$10K and more Gold for you F.r.e.e_Evaluation_NOW",
         "Hello Ayuba, I’m fine. How are you? I am at the office only on Thursdays."
         " I’ll be at the office on June 24 (Thursday) all day from 8 am to 5 pm."]

# convert these two messages to integers using Count Vectorized Technique
email_convert = vec.transform(emails)

# Predict the out come
predicted_model = model.predict(email_convert)
print(predicted_model)

# Measure my accuracy, first convert x_test into number
x_test_convert = vec.transform(x_test)
model_score = model.score(x_test_convert, y_test)
print(model_score)

"""The code above works perfect, but you can see that anytime i want to use my x_train, i have to convert to a number,
this can be done using a function in sklearn call Pipeline, mesaning you dont need to convert the x_train to number
 before running the code. Below is how it works"""

# Creating pipeline classifier
pipe_classifier = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("Naive_bayes", MultinomialNB())
])

# Training the model with pipeline
pipe_classifier.fit(x_train, y_train)

# Prediction with pipeline
predicted_model = pipe_classifier.predict(emails)
print(predicted_model)

# Checking the score with pipeline
model_score = pipe_classifier.score(x_test, y_test)
print(model_score)