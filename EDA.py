# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:33:56 2020

@author: Ifara
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

employees_data =  pd.ExcelFile('employees_data.xlsx')

existing_employees = pd.read_excel(employees_data, 'Existing employees')
employees_left = pd.read_excel(employees_data, 'Employees who have left')

#EMPLOYEES WHO LEFT
print('  Data overview of employees who have left\n','-'*45)
print(employees_left.info())
print('  Statistics\n','-'*45)
print(employees_left.describe())

#department visualisation
print('  Department information\n','-'*45)
print(employees_left['dept'].value_counts())
department = employees_left['dept'].value_counts()
x = department.index.to_list()
y = department.to_list()
plt.figure()
plt.title('Department of employees who left')
plt.xlabel('Department')
plt.ylabel('Employee count')
plt.bar(x, y)
plt.savefig('employees_left_department')
plt.show()

#salary visualisation
print('  Salary information\n','-'*45)
print(employees_left['salary'].value_counts())
salary = employees_left['salary'].value_counts()
x = salary.index.to_list()
y = salary.to_list()
plt.figure()
plt.title('Salary of employees who left')
plt.xlabel('salary')
plt.ylabel('Employee count')
plt.bar(x, y)
plt.savefig('employees_left_salary')
plt.show()

#visualisation of numeric features
employees_left.hist(figsize=(20,20))
plt.savefig('employees_left_hist')
plt.show()

#EXISTING EMPLOYEES
print('  Data overview of existing employees\n','-'*45)
print(existing_employees.info())
print('  Statistics\n','-'*45)
print(existing_employees.describe())

#department visualisation
print('  Department information\n','-'*45)
print(existing_employees['dept'].value_counts())
department = existing_employees['dept'].value_counts()
x = department.index.to_list()
y = department.to_list()
plt.figure()
plt.title('Department of existing employees')
plt.xlabel('Department')
plt.ylabel('Employee count')
plt.bar(x, y)
plt.savefig('existing_employees_salary')
plt.show()

#salary visualisation
print('  Salary information\n','-'*45)
print(existing_employees['salary'].value_counts())
salary = existing_employees['salary'].value_counts()
x = salary.index.to_list()
y = salary.to_list()
plt.figure()
plt.title('Salary of existing employees')
plt.xlabel('salary')
plt.ylabel('Employee count')
plt.bar(x, y)
plt.savefig('existing_employees_salary')
plt.show()

#visualisation of numeric features
existing_employees.hist(figsize=(20,20))
plt.savefig('existing_employees_hist')
plt.show()

#percentage of employees who left from each department
left = employees_left['dept'].value_counts()
columns = left.index.to_list()
left = left.to_list()
total_employees_dept = existing_employees['dept'].append(employees_left['dept'], ignore_index=True)
total_employees_dept = total_employees_dept.value_counts()
total = total_employees_dept.to_list()
percentage = []
for i in range(10):
    percent = round(((left[i]/total[i])*100),2)
    percentage.append(percent)
    
dept_percentage = pd.Series(data=percentage, index=columns)
x = columns
y = percentage
plt.figure(figsize=(12,7))
plt.title('Percentage of employees who left from each department')
plt.xlabel('department')
plt.ylabel('percentage')
plt.bar(x, y)
plt.savefig('employees_left_dept_percentage')
plt.show()
