#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 21:39:37 2019

@author: chengcheng
"""
##first draft:
#those who don't have dateofBirth have no age;
#those who have missing value in 'mother'; 'father';'heir';'spouse';also have no value
#in 'isAliveMother', 'isAliveFather','isAliveHeir', 'isAliveSpouse'
#assuing all the values are correct( e.g. male chase mentioned in class..?)

#cc-startegy1: replace those who have m/f/heir with 1,those miss these value w/ 0
#Q: why missing? 'real missing' or 'just missing somehow'
#then fill missing in any 'isAlive' with 2

#remain to tackle: culture; house; title; date of birth or age has very low corr.
ds['mother2'] = ds['mother'].apply(lambda x: 1 if not pd.isnull(x) else 0)
ds['isAliveMother'] = ds['isAliveMother'].fillna(2)
ds['isAlive'].corr(ds['isAliveHeir'])

###experimenting:
import seaborn as sns

fig, ax = plt.subplots(figsize = (15, 3)) 
   
plt.subplot(1, 3, 1)
sns.countplot(x='m_culture', hue='isAlive', data=ds, palette='RdBu') 
plt.xticks([0,1], ['No', 'Yes']) 

plt.subplot(1, 3, 2)
sns.countplot(x='m_title', hue='isAlive', data=ds, palette='RdBu') 
plt.xticks([0,1], ['No', 'Yes']) 


plt.subplot(1, 3, 3)
sns.countplot(x='m_house', hue='isAlive', data=ds, palette='RdBu') 
plt.xticks([0,1], ['No', 'Yes']) 
plt.show()
######
fig, ax = plt.subplots(figsize = (9, 6))

plt.subplot(2, 2, 1)
sns.countplot(x='m_spouse', hue='isAlive', data=ds, palette='RdBu') 
plt.xticks([0,1], ['No', 'Yes']) 

plt.subplot(2, 2, 2)
sns.countplot(x='m_mother', hue='isAlive', data=ds, palette='RdBu') 
plt.xticks([0,1], ['No', 'Yes']) 

plt.subplot(2, 2, 3)
sns.countplot(x='m_father', hue='isAlive', data=ds, palette='RdBu') 
plt.xticks([0,1], ['No', 'Yes']) 

plt.subplot(2, 2, 4)
sns.countplot(x='m_heir', hue='isAlive', data=ds, palette='RdBu') 
plt.xticks([0,1], ['No', 'Yes']) 

plt.tight_layout()
plt.show()
#Except 'name', all the 7 other text columns have missing values. 
##############################################
ds_died=ds.loc[ds['isAlive'] == 0]
ds_died.describe().round(2)
ds_na=ds.loc[ds['age'].isnull() ]
ds_na.describe().round(2)
####before flagging, can actully drop na to see the distribution
        #and decide the imputing strategy: e.g. substitute w/
        #0/median/mean...
################
import plotly
import plotly.graph_objs as go
from plotly import tools
trace_Hist = []
all_columns = list(ds.columns.values)

for i in all_columns:
    trace_Hist.append(go.Histogram(x = ds[i]))
    
fig_Hist = tools.make_subplots(rows = 5, cols = 7, subplot_titles=(all_columns))
rows_Hist = list(range(1,5))
columns_Hist = list(range(1,7)) 
intial_Hist = 0

for i in rows_Hist:
    for j in columns_Hist:
        fig_Hist.append_trace(trace_Hist[intial_Hist],i,j)
        intial_Hist += 1

plotly.offline.plot(fig_Hist, filename='Histogram')
#
trace_Histogram = []
all_columns = list(ds_died.columns.values)

for i in all_columns:
    trace_Histogram.append(go.Histogram(x = ds[i]))
    
fig_Histogram = tools.make_subplots(rows = 5, cols = 7, subplot_titles=(all_columns))
rows_Histogram = list(range(1,5))
columns_Histogram = list(range(1,7)) 
intial_Histogram = 0

for i in rows_Histogram:
    for j in columns_Histogram:
        fig_Histogram.append_trace(trace_Histogram[intial_Histogram],i,j)
        intial_Histogram += 1

plotly.offline.plot(fig_Histogram, filename='Histogram')
#####鸡肋：
import seaborn as sns
df_corr = ds.corr().round(2)
sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize=(15,15))
df_corr2 = df_corr.iloc[1:18, 1:18]
sns.heatmap(df_corr2,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)

###unused experiment
fill = 2
ds['isAliveHeir'] = ds['isAliveHeir'].fillna(fill)

ds.loc[ds.title == 'Ser', 'male'] = 1
print(ds.isnull().sum())


for i in (0,1945):
    if ds.loc[i,'house']=='House Targaryen':
        ds.loc[i,'house']=1
    else:
        ds.loc[i,'house']=0
        
                
fill = ds['dateOfBirth'].median()
ds['dateOfBirth'] = ds['dateOfBirth'].fillna(fill)
fill = 'unknown'
ds['title'] = ds['title'].fillna(fill)
fill = 'unknown'
fill = 'n/a'
ds['mother'] = ds['mother'].fillna(fill)
fill = 'n/a'
ds['father'] = ds['father'].fillna(fill)
fill = 'n/a'
ds['heir'] = ds['heir'].fillna(fill)
fill = 'n/a'
ds['spouse'] = ds['spouse'].fillna(fill)


fill = ds['isAliveMother'].median()
ds['isAliveMother'] = ds['isAliveMother'].fillna(fill)
fill = ds['isAliveFather'].median()
ds['isAliveFather'] = ds['isAliveFather'].fillna(fill)
fill = ds['isAliveHeir'].median()
ds['isAliveHeir'] = ds['isAliveHeir'].fillna(fill)
fill = ds['isAliveSpouse'].median()
ds['isAliveSpouse'] = ds['isAliveSpouse'].fillna(fill)

#get dummy variables, get the correlation between each type of title and 'male'
#use this result as benchmark to impute any possible misclassification of title under 'male'
for column in df:
    print(df[column])
    
dum = pd.get_dummies(ds['title'])
df = pd.concat([ds['male'], dum], axis=1) 
dcorr=df.corr().round(2)

for col in dcorr.columns:
    if dcorr.loc['male',col]>0:
        ds.loc[ds.title == col, 'male'] = 1
    if dcorr.loc['male',col]<0:
        ds.loc[ds.title == col, 'male'] = 0
               
dcorr=ds.corr().round(2)   

       
_dummies = pd.get_dummies(list(diamonds['channel']))
print(channel_dummies)
results = lm_full.fit()
print(results.summary())

ds_dm=pd.get_dummies(ds)
print(ds_dm.isnull().sum())
#####
ds_data=ds.drop(['name','title','mother','father','heir','spouse','culture',
                 'dateOfBirth','house','age'                                                   
                           ],axis=1)
ds_target=ds.loc[:,'isAlive']

from sklearn.model_selection import train_test_split 
import statsmodels.formula.api as smf
X_train, X_test, y_train, y_test = train_test_split(ds_data,
                                                    ds_target,
                                                    test_size = 0.2,
                                                    random_state = 520,
                                                    stratify=ds_target)

ds_train=pd.concat([X_train, y_train], axis = 1)
ds_train.corr().round(2)
print(ds_train.isnull().sum())

#creating outlier flag:set the hi/lo threshold 
housing['out_lot_area'] = 0

#using loop
for val in enumerate(housing.loc[ : , 'Lot Area']):
    
    if val[1] >= lot_area_hi:
        housing.loc[val[0], 'lot_area_hi'] = 1
#or use a lambda function       
housing['out_lot_area'] = housing['Lot Area'].apply(lambda val: 1 if val < lot_area_hi else 0)        


####before flagging, can actully drop na to see the distribution
        #and decide the imputing strategy: e.g. substitute w/
        #0/median/mean...
################
from sklearn.linear_model import Lasso 
lasso = Lasso(alpha=.01) 
reg=lasso.fit(X,y) 
lasso_coef =lasso.coef_ 
print(lasso_coef)   
# Plot the coefficients 
plt.plot(range(len(X.columns)), lasso_coef) 
plt.xticks(range(len(X.columns)), X.columns.values, rotation=60) 
plt.margins(0.02) 
plt.show() 
#####
from sklearn.linear_model import Ridge
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,
                                               random_state=42)
ridge=Ridge(alpha=.1,normalize=True)
ridge.fit(X_train,y_train)
ridge_pred=ridge.predict(X_test)
ridge.score(X_test,y_test)
####
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn import linear_model 
reg = linear_model.LinearRegression() 
reg.fit(X_train,y_train) 
y_pred =reg.predict(X_test) 
print("R^2: {}".format(reg.score(X_test, y_test))) 

from sklearn.metrics import mean_squared_error 
rmse = np.sqrt(mean_squared_error(y_test,y_pred)) 
print("Root Mean Squared Error: {}".format(rmse)) 

from sklearn.model_selection import cross_val_score 
cv_scores = cross_val_score(reg,X,y,cv=5) 
print(cv_scores) 
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores))) 

# Import the necessary modules 
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline 
from sklearn.neighbors import KNeighborsClassifier
# Setup the pipeline steps: steps 
steps = [('scaler', StandardScaler()), 
        ('knn', KNeighborsClassifier())] 
# Create the pipeline: pipeline 
pipeline = Pipeline(steps) 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.1,random_state=508) 
knn_scaled = pipeline.fit(X_train,y_train) 
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train) 

print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test,y_test))) 
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test,y_test))) 

# Import the necessary modules 
from sklearn.svm import SVC
steps = [('scaler', StandardScaler()), 
         ('SVM', SVC())] 
pipeline = Pipeline(steps)  
parameters = {'SVM__C':[1, 10, 100], 
              'SVM__gamma':[0.1, 0.01]} 
# Instantiate the GridSearchCV object: cv 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# Creating a hyperparameter grid
estimator_space = pd.np.arange(100, 1350, 250)
criterion_space = ['gini', 'entropy']
param_grid = {'n_estimators' : estimator_space,
              'criterion' : criterion_space}

full_forest_grid = RandomForestClassifier(max_depth = None,
                                          random_state = 508)

full_forest_cv = GridSearchCV(full_forest_grid, param_grid, cv = 3)
full_forest_cv.fit(X_train,y_train)
y_pred_prob=full_forest_cv.predict_proba(X_test)[:,1]
roc_auc_score(y_test,y_pred_prob)




#########################
from sklearn.linear_model import LogisticRegression

logreg_fit = logreg.fit(X_train, y_train)
logreg_pred = logreg_fit.predict(X_test)

c_space = np.logspace(-5, 8, 15) 
param_grid = {'C': c_space} 

logreg = LogisticRegression() 
logreg_cv = GridSearchCV(logreg, param_grid, cv=5) 
logreg_cv.fit(X,y) 
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))  
print("Best score is {}".format(logreg_cv.best_score_)) 

# Let's compare the testing score to the training score.
print('Training Score', logreg_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_fit.score(X_test, y_test).round(4))

# Import necessary modules 
from sklearn.multiclass import OneVsRestClassifier
clf = OneVsRestClassifier(LogisticRegression()) 
clf.fit(X_train, y_train) 
y_pred_prob=clf.predict_proba(X_test)[:,1]
roc_auc_score(y_test,y_pred_prob) 

#AUC using cross-validation
from sklearn.model_selection import cross_val_score
cv_score=cross_val_score(logreg,X,y,cv=5,scoring='roc_auc')
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_score)) 

####run to check the 50 numvers .logsapce generate###
c=np.logspace(-3,1000)
param_grid = {'C': c} 
####other model
from sklearn.metrics import roc_auc_score
model=___()
model.fit(X_train,y_train)
y_pred_prob=___.predict_proba(X_test)[:,1]


#############################
########other models#########
#############################

##########################################################
####   tuning hyperparameter C in logreg   ###############
##########################################################
from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import LogisticRegression 

param={'C': [0.1,1,10]}##default is 1.
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict_proba

lr = LogisticRegression() 
lr_cv = GridSearchCV(lr, param) 
lr_cv.fit(X_train,y_train)

print("Tuned Logistic Regression Parameters: {}".format(lr_cv.best_params_))  
print("Best score is {}".format(lr_cv.score(X_test,y_test))) 
#####   try the best C in logreg######
lr = LogisticRegression(C=.1) 

lr.fit(X_train,y_train)

y_pred_prob=lr.predict_proba(X_test)[:,1]

roc_auc_score(y_test,y_pred_prob)#0.833



#######################################
#############    KNN      #############
#######################################

# Running the neighbor optimization code with a small adjustment for classification
from sklearn.neighbors import KNeighborsClassifier

neighbors_settings = range(1,51)

training_accuracy = []

test_accuracy = []

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))

fig, ax = plt.subplots(figsize=(10,7))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

#####use the optimal n_neighbors######
knn_clf = KNeighborsClassifier(n_neighbors = 9)

knn_clf_fit = knn_clf.fit(X_train, y_train)

y_pred_prob=knn_clf.predict_proba(X_test)[:,1]

roc_auc_score(y_test,y_pred_prob)##0.805

pred_y=knn_clf.predict(X_test)

print('Training Score', knn_clf_fit.score(X_train, y_train).round(4))

print('Testing Score:', knn_clf_fit.score(X_test, y_test).round(4))

print(classification_report(y_true = y_test,
                            y_pred = pred_y))

print(confusion_matrix(y_test,pred_y))
#####knn got worse AUC score than my best logreg, but seemingly better accuracy 
###i searched, the overall accuracy scores seem to be based on some cutpoints
#while AUC takes into consideration all possible 'points'....

##############################################
#########  random forest classifier  #########

from sklearn.ensemble import RandomForestClassifier

full_forest_gini = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'gini',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,####
                                     random_state = 508)

full_gini_fit = full_forest_gini.fit(X_train, y_train)
# Scoring the gini model
print('Training Score', full_gini_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_gini_fit.score(X_test, y_test).round(4))

y_pred=full_forest_gini.predict(X_test)
y_pred_prob=full_gini_fit.predict_proba(X_test)[:,1]##.predict_proba method can
#be used upon either the model or the fitted model.
y_pred_prob=full_forest_gini.predict_proba(X_test)[:,1]
roc_auc_score(y_test,y_pred_prob)##.845

pred_y=full_forest_gini.predict(X_test)

print(confusion_matrix(y_test,pred_y))
print('MAE:', metrics.mean_absolute_error(y_test,pred_y))
print('MSE:',metrics.mean_squared_error(y_test,pred_y))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,pred_y)))

np.sqrt(metrics.mean_squared_error(y_test,pred_y))/(np.mean(y))


####################################################################
######Tuning hyperparameters#####random forest classifier##########
###################################################################
from sklearn.model_selection import GridSearchCV

# hyperparameter that have been separately tried:
bootstrap_space = [True, False]
warm_start_space = [True, False]
criterion_space = ['gini', 'entropy']
####creating hyperparameter grid:
estimator_space = pd.np.arange(250, 1100, 250)
leaf_space = pd.np.arange(5, 150, 15)

param_grid = {'n_estimators' : estimator_space,
              'min_samples_leaf' : leaf_space}


full_forest_gini = RandomForestClassifier(bootstrap=False,
                                          warm_start=True,
                                          random_state = 508,
                                          criterion='gini')

full_forest_cv = GridSearchCV(full_forest_grid, param_grid)

full_forest_cv.fit(X_train, y_train)
# Print the optimal parameters and best score
print("Tuned Parameter:", full_forest_cv.best_params_)
print("Tuned Accuracy:", full_forest_cv.best_score_.round(4))





##########MY Logistic regression ###################
#################compute AUC#######################
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import scale 

X_scaled = scale(X) 

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=.1,
                                               random_state=508)

logreg=LogisticRegression()

logreg.fit(X_train,y_train)

y_pred_prob=logreg.predict_proba(X_test)[:,1]

roc_auc_score(y_test,y_pred_prob)#0.834

#####  r square  ######
logreg.score(X_test,y_test)
########################
# Creating a confusion matrix
########################
import seaborn as sns
from sklearn.metrics import confusion_matrix

pred_y=logreg.predict(X_test)

print(confusion_matrix(y_test,pred_y))

labels = ['Not Alive', 'Alive']

cm = confusion_matrix(y_test,pred_y)

sns.heatmap(cm,
            annot = True,
            xticklabels = labels,
            yticklabels = labels,
            cmap = 'pink_r')            
            
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of the classifier')
plt.show()

########################
# Creating a classification report
########################
from sklearn.metrics import classification_report

print(classification_report(y_true = y_test,
                            y_pred = pred_y))
# Changing the labels on the classification report
print(classification_report(y_true = y_test,
                            y_pred = pred_y,
                            target_names = labels))
######draw a roc###########
from sklearn.metrics import roc_curve

y_pred_prob=logreg.predict_proba(X_test)[:,1]

fpr,tpr,thresholds=roc_curve(y_test,y_pred_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

#####othrt accuracy metrics#####
from sklean import metrics

print('MAE:', metrics.mean_absolute_error(y_test,pred_y))
print('MSE:',metrics.mean_squared_error(y_test,pred_y))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,pred_y)))

np.sqrt(metrics.mean_squared_error(y_test,pred_y))/(np.mean(y))#.584

#was thinking about save the feature engineered dataset to excel to save time
#every time dealng w/ it before running the model
ds.to_excel('got1.xlsx')

###############DecisionTreeClassifier################
#####################################################
criterion_space = ['gini', 'entropy']
depth_space = pd.np.arange(5, 11)
leaf_space = pd.np.arange(15, 51)
param_grid = {'max_depth' : depth_space,
              'min_samples_leaf' : leaf_space,
              'criterion': criterion_space}

# Building the model object one more time
tree = DecisionTreeClassifier(random_state = 508)

tree_cv = GridSearchCV(tree, param_grid, cv = 3)

tree_cv.fit(X_train, y_train)

print("Tuned Parameter:", tree_cv.best_params_)
print("Tuned Accuracy:", tree_cv.best_score_.round(4))

####using the optimal parameters####
c_tree_optimal = DecisionTreeClassifier(criterion = 'entropy',
                                        random_state = 508,
                                        max_depth = 5,
                                        min_samples_leaf = 18)

c_tree_optimal_fit = c_tree_optimal.fit(X_train, y_train)
y_pred=c_tree_optimal.predict(X_test)
y_pred_prob=c_tree_optimal.predict_proba(X_test)[:,1]
roc_auc_score(y_test,y_pred_prob)#0.8037

########using RandomizedSearchCV##############
 from scipy.stats import randint 
from sklearn.tree import DecisionTreeClassifier  
from sklearn.model_selection import RandomizedSearchCV 

param_dist = {"max_depth": pd.np.arange(5, 11), 
              "max_features": randint(5, 10), 
              "min_samples_leaf": randint(15, 50), 
              "criterion": ["gini", "entropy"]} 

  
tree = DecisionTreeClassifier() 
tree_cv = RandomizedSearchCV(tree, param_dist, cv=3) 
tree_cv.fit(X,y) 

print(tree_cv.best_params_)
print(tree_cv.best_score_)

c_tree_optimal = DecisionTreeClassifier(criterion = 'gini',
                                        random_state = 508,
                                        max_depth = 5,
                                        max_features=9,
                                        min_samples_leaf=15)

c_tree_optimal_fit = c_tree_optimal.fit(X_train, y_train)
y_pred=c_tree_optimal.predict(X_test)
y_pred_prob=c_tree_optimal.predict_proba(X_test)[:,1]
roc_auc_score(y_test,y_pred_prob)##.8253



