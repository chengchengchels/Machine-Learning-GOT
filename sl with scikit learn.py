#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 18:16:00 2019

@author: chengcheng
"""

from sklearn import datasets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

iris=datasets.load_iris()

type(iris)
print(iris.keys())
type(iris.target)
type(iris.data)
iris.data.shape

#################exploratory data analysis(EDA)###########################
#assignning value
X=iris.data
y=iris.target
#build dataframe
df=pd.DataFrame(X,columns=iris.feature_names)

pd.set_option('display.expand_frame_repr', False)
print(df.head())

#pd.set_option('display.max_columns', None)  
#pd.set_option('max_colwidth', -1)



################visual EDA#################################################

ve=pd.scatter_matrix(df,c=y,figsize=[8,8],s=150,marker='D')
#pass our df along with our target variable as argument to the parameter c, 
#which stands for column
#also, pass a list to figsize, which specifies for us the size of figure
#as well as marker size and shape(s)
#the result is a matrix of figures,on the diagonal  
#the off-diagonal figures are scatter plots of column features vs row features
#colored by the target variables. 




###############to fit a classifier########################################

from sklearn.neighbors import KNeighborsClassifier#case sensitive
knn=KNeighborsClassifier(n_neighbors=6)#Instantiate to a classifier
#set the numver of neighbors to 6 and assign it to variable knn
knn.fit(iris['data'],iris['target'])# fit the classifier to trainning set-
#-the labled data: apply method fit() and pass it 2 arguments: feature and target 
#numpy arrays: contiunous/(vs cateigorical , missing data...)
#(row: obervations; column: features)

iris['data'].shape
iris['target'].shape
#target needs to be a single column with the same number of observation as 
#feature data

########predict on unlabled data#########################################

## Predict the labels for the training data X
#y_pred = knn.predict(X)

prediction=knn.predict(X_new)
X_new.shape

print('Prediction {}'.format(prediction))

##########measure model performance#####################################

###train/test split:
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.1,
                                                   random_state=21)


knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print("Test set prediction:\n {}". format(y_pred))

#check out accuracy of models: score method
knn.score(X_test,y_test)


#######overfitting and underfitting####################################

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train,y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test,y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

#######regression#######################################################
########################################################################

#load the data:
boston=pd.read_csv('boston.csv')
print(boston.head())
#creating feature and target arrays:
x=boston.drop('MEDV',axis=1).values#drop the target
y=boston['MEDV'].values#keep only the target: .values attribute returns numpy arrays

#predicting house value from a single feature

X_rooms=X[:,5]#the 5th column
type(X_rooms), type(y)

y=y.reshape(-1,1)

X_rooms=X_rooms,reshape(-1,1)


#plotting house value vs. number of rooms
plt.scatter(X_rooms,y)
plt.ylabel('Value of house/1000($)')
plt.xlabel('Number of rooms')
plt.show()

#fitting a regression model
import numpy as np
from sklearn import linear_model
reg=linear_model.LinearRegression()
reg.fit(X_rooms,y)
prediction_space=np.linspace(min(X_rooms),
                             max(X_rooms)).reshape(-1,1)

plt.scatter(X_rooms,y,color='blue')

plt.plot(prediction_space,reg.predict(prediction_space),
         color='black',linewidth=3)
plt.show()


#exploring data:
df.info()
df.describe()
df.head()
df.corr()

##########linear regression on all features##############
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=.3,
                                                random_state=42)
reg_all=linear_model.LinearRegression()
reg_all.fit(X_train,y_train)#fit on the trainning set
y_pred=reg_all.predict(X_test)#predict on the test set

#compute r-square:
reg_all.score(X_test,y_test)

##############cross_validation in scikit-learn##########
from sklearn.model_selection import cross_val_score 
#instantiate a model 
reg=linear_model.LinearRegression()
#call cross_val_score on the regressor, feature data and the target data
#and number of fold with cv
cv_results=cross_val_score(reg,X,y,cv=5)

print(cv_results)
np.mean(cv_results)


###########ridge regression#############################
from sklearn.linear_model import Ridge
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,
                                               random_state=42)
ridge=Ridge(alpha=.1,normalize=True)
ridge.fit(X_train,y_train)
ridge_pred=eidge.predict(X_test)
ridge.score(X_test,y_test)

###########lasso regression#############################
...

lasso=Lasso(alpha=.1,normalize=True)
lasso.fit(X_train,y_train)
lasso_pred=lasso.predict
lasso.score(X_test,y_test)

#feature selection
from sklearn.linear_model import Lasso
names=boston.drop('MEDV',axis=1).columns
lasso=Lasso(alpha=.1)
lasso_coef=lasso.fit(X,y).coef_
plt.plot(range(len(names)),lasso_coef)
plt.xticks(range(len(names)),names,rotation=60)
plt.ylabel('Coefficients')
plt.show()


#####accuracy of your model######
#condusino matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
knn=KNeighborsClassifier(n_neighbors=8)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.4,
                                               random_state=42)

knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

##another Classification model：logistic regression and the ROC curve##########
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
logreg=LogisticRegression()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.4,
                                               random_state=42)
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
#probability thresholds, e.g. 0.5
#the set of points we get when trying all possible thresholds is called the
#receiver operating characteristic curve( ROC curve)
#plot the roc curve:



#compute AUC
from sklearn.metrics import roc_auc_score
logreg=LogisticRegression()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.4,
                                               random_state=42)
logreg.fit(X_train,y_train)
y_pred_prob=logreg.predict_proba(X_test)[:,1]

roc_auc_score(y_test,y_pred_prob)
#AUC using cross-validation
from sklearn.model_selection import cross_val_score
cv_score=cross_val_score(logreg,X,y,cv=5,scoring='roc_auc')

#hyperparameter tuning: grid search cross_validation 
#GridSearchCV in scikit-learn
from sklearn.model_selection import GridSearchCV

param_grid={'n_neighbors':np.arange(1,50)}
#specify the hyperparameter as a dictionary containing a list of value we wish
#to tune the relevant hyperparameter over.
knn=KNeighborsClassifier()
knn_cv=GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(X,y)

knn_cv.best_params_
knn_cv.best_score_



#######################
#preprocessing data
#######################

import pandas as pd
df=pd.read_csv('auto.csv')
df_origin=pd.get_dummies(df)
print(df_origin.head())

df_origin=df.origin.drop('origin_Asia', axis=1)

#linear regression with dummy variables
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
X_train,X_test,y_train, y_test=train_test_split(X,y, test_size=.3,random_state_42)
ridge=Ridge(alpha=.5,normalized=True).fit(X_train,y_train)
ridge.score(X_test,y_test)


############################
#handling missing data
############################
df=pd.read_csv('diabetes.csv')
df.info()
print(df.head())

#dropping missing data
df.col_name.replace(0,np.nan,inplace=True)
df=df.dropna()
df.shape
#imputing missing data: make educated guess
#using mean:
from sklearn.preprocessing import Imputer
imp=Imputer(missing_value='NaN', strategy='mean',axis=0)#mean impute by column,axis=1 means by row
      
imp.fit(X)
X=imp.transform(X)

#imputing with a pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
logreg=LogisticRegression()
steps=[('imputation',imp),('logistic_regression',logreg)]
pipeline=Pipeline(steps)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=42)
pipeline.fit(X_train,y_train)
y_pred=pipeline_predict(X_test)
pipeline.score(X_test,y_test)

######imputing missing data in a ML Pipeline I
# Import the Imputer module
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC

# Setup the Imputation transformer: imp
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
        ('SVM', clf)]


######imputing missing data in a ML Pipeline II
# Import necessary modules
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
        ('SVM', SVC())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3,random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train,y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test,y_pred))

########################
#centering and scaling#
#######################
from sklearn.preprocessing import scale
x_scaled=scale(X)
np.mean(X),np.std(X)
np.mean(X_scaled), np.std(X_scaled)
#scaling in a ppeline
from sklean.preprocessing import StandardScaler

steps=[('scaler', StandardScaler()),
       ('knn',KNeighborClassifier())]

pipeline=Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=21)

knn_scaled=pipeline.fit(X_train,y_train)
y_pred=pipeline.predict(X_test)
accuracy_score(y_test,y_pred)

knn_unscaled=KNeighborClassifier().fit(X_train,y_train)

knn_unscaled.score(X_test,y_test)

#CV and scaling in a pipeline
#build pipeline
steps=[('scaler', StandardScaler()),
       ('knn',KNeighborClassifier())]
pipeline=Pipeline(steps)
#specify parameters by creating a dictionary
parameters={knn__n_neighbors: np.arange(1,50)}

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=21)

#perform grid search
cv=GridSearchCV(pipeline,param_grid=parameters)
cv.fit(X_train,y_train)


y_pred=cv.predict(X_test)#test=hold out set
print(cv.best_params_)
print(cv.score(X_test,y_test))



