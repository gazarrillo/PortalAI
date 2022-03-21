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

def getXMLValues(fileName):
    # get individual values from xml
    revenue = fileName.getElementsByTagName(TOTAL_REVENUE)
    expenses = fileName.getElementsByTagName(TOTAL_EXPENSES)
    salaries = fileName.getElementsByTagName(SALARIES)

    return revenue,expenses, salaries

def addXMLValues(rev,exp,sal):
    indvArr = []
    addXMLValue(indvArr,rev)
    addXMLValue(indvArr,exp)
    addXMLValue(indvArr,sal)
    return indvArr

def addXMLValue(arr, val):
    arr.append(float(val[0].firstChild.data))

def addOutput(exp,rev):
    return float(exp[0].firstChild.data)/float(rev[0].firstChild.data)

#get inputs and outputs from all xml files
def trainModel():
    for i in range(TOTAL_TRAINING_FILES):

        file = minidom.parse(getFileName(i))

        revenue, expenses, salaries = getXMLValues(file)

        #add data to input array
        inputs.append(addXMLValues(revenue,expenses,salaries))

        #calculate and add data to output array
        outputs.append(addOutput(expenses, revenue))

def testXML(fileName):
    file = minidom.parse(fileName)
    revenue, expenses, salaries = getXMLValues(file)
    test_inputs.append(addXMLValues(revenue,expenses,salaries))
    test_outputs.append(addOutput(expenses, revenue))

trainModel()
testXML('PortalXML/TestXML21.xml')

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