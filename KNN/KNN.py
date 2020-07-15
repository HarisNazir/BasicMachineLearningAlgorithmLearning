import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

#Read In Data
data = pd.read_csv("car.data")
print(data.head)

#Our dataset uses strings instead of numerical values, therefore we will have to do preprocessing before training a model

preprocess = preprocessing.LabelEncoder()
buying = preprocess.fit_transform(list(data["buying"])) #Everything inside buying column will be transformed into a numerical value
maint = preprocess.fit_transform(list(data["maint"]))
door = preprocess.fit_transform(list(data["door"]))
persons = preprocess.fit_transform(list(data["persons"]))
lug_boot = preprocess.fit_transform(list(data["lug_boot"]))
safety = preprocess.fit_transform(list(data["safety"]))
cls = preprocess.fit_transform(list(data["class"]))

predict = "class"

#X is features and Y is label
x = list(zip(buying, maint, door, persons, lug_boot, safety)) #Zip is putting all attributes into one large list
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#neighbours = int(input("How many neighbours would you like: "))
model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)

print(accuracy)
#print(accuracy * 100, "%")

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])