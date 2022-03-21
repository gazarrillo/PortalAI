import numpy as np
from sklearn.linear_model import LinearRegression
from xml.dom import minidom
from helpers import *

TOTAL_REVENUE = "CYTotalRevenueAmt"
TOTAL_EXPENSES = "CYTotalExpensesAmt"
SALARIES = "CYSalariesCompEmpBnftPaidAmt"
TOTAL_TRAINING_FILES = 20

inputs = []
outputs = []

test_inputs= []
test_outputs = []

#get inputs and outputs from all xml files
for i in range(TOTAL_TRAINING_FILES+1):

    file = minidom.parse(getFileName(i))

    #get individual values from xml
    revenue = file.getElementsByTagName(TOTAL_REVENUE)
    expenses = file.getElementsByTagName(TOTAL_EXPENSES)
    salaries = file.getElementsByTagName(SALARIES)

    #add data to input array
    indvArr = []
    indvArr.append(float(revenue[0].firstChild.data))
    indvArr.append(float(expenses[0].firstChild.data))
    indvArr.append(float(salaries[0].firstChild.data))

    #calculate and add data to output array
    output = float(expenses[0].firstChild.data)/float(revenue[0].firstChild.data)

    if (i == 20):
        test_inputs.append(indvArr)
        test_outputs.append(output)
    else:
        inputs.append(indvArr)
        outputs.append(output)

x, y = np.array(inputs), np.array(outputs)
x_new = np.array(test_inputs)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
y_pred = model.intercept_ + np.sum(model.coef_ * x, axis=1)
y_new = model.predict(x_new)

print("!!!!!    TRAINING DATASET   !!!!!")
print("[INPUTS]:")
print(x)
print("[OUTPUTS]:")
print(y)
print("[RELATION FACTOR]:")
print(r_sq)
#print(y_pred)
print("!!!!!    TESTING DATASET   !!!!!")
print("[PREDICTED OUTPUT]:")
print(y_new)
print("[ACTUAL OUTPUT]:")
print(test_outputs)