# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:43:14 2017

@author: pulki
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('battery_log.txt' , sep = ',' , names = ['c_time' , 'u_time'] ) 

X = dataset.iloc[:, :1].values
y = dataset.iloc[:, 1:].values




#decision tree regression
from sklearn.tree import DecisionTreeRegressor
regressor_d = DecisionTreeRegressor(random_state=0)
regressor_d.fit(X,y)

#random forest
from sklearn.ensemble import RandomForestRegressor
regressor_r = RandomForestRegressor(n_estimators = 500 , random_state=0)
regressor_r.fit( X ,y )



from sklearn.linear_model import LinearRegression
regressor_p = LinearRegression()
regressor_p.fit(X_poly , y)
#y_pred = regressor.predict(2.81)




X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor_d.predict(X_grid), color = 'blue')
plt.title('decision tree regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()




X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor_r.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

y_pred1 = regressor_d.predict(2.81)
#y_pred2 = regressor_p.predict(pol2.81)
y_pred3 = regressor_r.predict(2.81)

r1 = r2_score(y , regressor_d.predict(X))
r2 = r2_score(y , regressor_r.predict(X))