


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression




data = pd.read_csv('Salary_Data.csv') #Importing Data
X = data.iloc[:,:-1].values #This ommits the last coloumn
y = data.iloc[:,1].values #This gives us the last coloumn


"""Now remeber its upto you,how much data you want to feed in the train and split.
i took test_size=1/3, which means 10 out of 30 train values undergoes this process  """

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=1/3, random_state=0)


""" After splitting the data into training and testing sets,
 finally, the time is to train our algorithm. """


regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

"""  linear regression model basically 
finds the best value for the intercept and slope """

print(regressor.intercept_)
print(regressor.coef_)

"""Now compare the actual output values for X_test 
with the predicted values"""
df = pd.DataFrame({'Actual': y_test.flatten(),'Predicted': y_pred.flatten()})
df



plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='black', linewidth=2)
plt.xlabel('Years of Expreience')
plt.ylabel('Salary')
plt.show()


