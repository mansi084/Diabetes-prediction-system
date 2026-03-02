#1.import dependencies
#2.data collecton and analysis 
#3.data standarization
#4.train test split
#5.training the model
#6.model evaluation (accuracy score for both training and testing data)
#7.making a predictive system


#importing the dependies
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm      #support vector machine
from sklearn.metrics import accuracy_score

data = pd.read_csv("diabetes.csv")

#printing first 5 rows of the dataset
print(data.head())

print(data.shape)

print(data['Outcome'].value_counts()) #0- Non-diabetic   1- Diabetic

print(data.groupby('Outcome').mean())

#seperating data and labels
X = data.drop(columns = 'Outcome')
Y=data['Outcome']

#Data standardization

scaler = StandardScaler()
scaler.fit(X)
standardised_data = scaler.transform(X)
print(standardised_data)

X = standardised_data
Y=data['Outcome']

#spliting data for training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,stratify=Y,random_state=2)

"""test_size=0.2 → Splits the data so 20% is used for testing and 80% for training.
stratify=y → Makes sure both training and testing data have the same balance of classes 
(for example, equal proportion of diabetic and non-diabetic patients).
random_state=2 → Makes the data split repeat the same way every time you run the program."""

print(X.shape, X_train.shape, X_test.shape)

#training the model
classifier = svm.SVC(kernel = 'linear')

#training the support vector machine classifier
classifier.fit(X_train, Y_train)

#Model evaluation

#accuracy score

#accuracy on training set data 

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print("Accuracy score of the training data : ",training_data_accuracy)

#accuracy on training set data 

X_test_prediction = classifier.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print("Accuracy score of the testing data : ",testing_data_accuracy)

#in overfitting, training data accuracy >>>> test data accuracy

#making a predictive system

input_data = (0,162,76,56,100,53.2,0.759,25)

#changing input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the data as we are predicting for one instance 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardize the input data 
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")
