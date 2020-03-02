# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:52:01 2020

@author: Ifara
"""

#importing libraries
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix

#importing dataset
employees_data =  pd.ExcelFile('employees_data.xlsx')

#combinig dataset
existing_employees = pd.read_excel(employees_data, 'Existing employees')
left = np.zeros(11428, dtype=int)
existing_employees['left'] = left
                  
employees_left = pd.read_excel(employees_data, 'Employees who have left')
left = np.ones(3571, dtype=int)
employees_left['left'] = left

comb_data = employees_left.append(existing_employees, ignore_index=True)



x = comb_data.iloc[:, 1:-1] #independent variable
y = np.array(comb_data['left'].values, dtype=np.float) #dependent variable

#creating dummy variables
dummy_salary = pd.get_dummies(x['salary'], prefix='salary')
dummy_dept = pd.get_dummies(x['dept'], prefix='dept')

x = x.drop(columns=['dept', 'salary'])

x = x.join(dummy_salary)
x = x.join(dummy_dept)

#splitting of dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3, stratify=y)

#model initialisation
model = RandomForestClassifier()

#cross validation of the model
k_fold = KFold(n_splits=10, random_state=3, shuffle=True)
validation = cross_val_score(model, x_train, y_train, cv=k_fold, scoring='accuracy')
print('Cross Validation score = ',validation.mean(), '\n')

#fitting of the model on training dataset
model = model.fit(x_train, y_train)

#prediction
y_pred = model.predict(x_test)

#EVALUATION
#classification report
print('                  Classification Report', '\n', '-'*55)
print(classification_report(y_test, y_pred), '\n')
print('Accuracy Score = ', accuracy_score(y_test, y_pred), '\n')
print('ROC_AUC Score = ', roc_auc_score(y_test, y_pred))

#ROC Curve
roc_auc = roc_auc_score(y_test, y_pred)

false_positive, true_positive, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(false_positive, true_positive, label='Random Forest (area = %0.3f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.plot([0, 1], [0, 1],'r--')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_img')
plt.show()

#confusion matrix
model_matrix = confusion_matrix(y_pred, y_test, [1,0])
sn.heatmap(model_matrix, fmt='.2f', annot=True, xticklabels = ["Left", "Existing"] , yticklabels = ["Left", "Existig"] )
plt.ylabel('Actual class')
plt.xlabel('Predicted class')
plt.title('Random Forest')
plt.savefig('confusion_matrix_img')
plt.show()

#predicting probabilties
y_pred_prob = model.predict_proba(x_test)
y_pred_prob = y_pred_prob[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_prob)
print('\nPredicting Probabities: \n')
print('roc_auc_score score: %.3f' % roc_auc)

#feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = [x_train.columns[i] for i in indices]
plt.figure(figsize=(18, 13))
plt.title("Feature Importance")
plt.bar(range(x_train.shape[1]), importances[indices]) 
plt.xticks(range(x_train.shape[1]), feature_names, rotation=90)
plt.savefig('feature_importance')
plt.show()



