import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

#Read in Data
data = pd.read_csv("student-mat.csv", sep=";")

#Only want these attributes
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

#Predict this attribute - On the dataset G3 is final grade
predict = "G3"

#Putting all attributes in to X Array except the G3 Attribute
x = np.array(data.drop([predict], 1))

y = np.array(data[predict])

best=0
for i in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear_regression_model = linear_model.LinearRegression()

    #Fit Data to find the best fit line
    linear_regression_model.fit(x_train, y_train)

    #Return Accuracy of model
    accuracy = linear_regression_model.score(x_test, y_test)

    #A number between 0 and 1, the higher the better
    print(accuracy)
    print(accuracy * 100)

    if accuracy > best:
        best = accuracy
        #Saves Data Plots
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear_regression_model, f)

pickle_in = open("studentmodel.pickle", "rb")

linear_regression_model = pickle.load(pickle_in)

predictions = linear_regression_model.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

#Visualising the Data
p_variable = input("What would you like the comparison variable to be: ")
p= p_variable #Used to be G1 (for future reference)
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()