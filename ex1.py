# Kasper Eloranta, H274212, kasper.eloranta@tuni.fi
# Pattern Recognition and Machine Learning, DATA.ML.200
# Exercise 1 (course level test)
# Kaggle team name: Kasper Eloranta

import pickle
import numpy as np
import csv
from sklearn import neighbors, datasets

with open("training_x.dat", 'rb') as pickleFile:
    x_tr = pickle.load(pickleFile)

with open("training_y.dat", 'rb') as pickleFile:
    y_tr = pickle.load(pickleFile)

with open("validation_x.dat", 'rb') as pickleFile:
    x_val = pickle.load(pickleFile)

x_tr_prepared = np.zeros((len(x_tr),64))
x_val_prepared = np.zeros((len(x_val),64))

# preparing training and test data, ugly but effective solution
imgindex = 0
pixindex = 0
for i in x_tr:
    for j in i:
        for k in j:
            x_tr_prepared[imgindex][pixindex] = k[0]
            pixindex += 1
    imgindex += 1
    pixindex = 0

imgindex = 0
pixindex = 0
for i in x_val:
    for j in i:
        for k in j:
            x_val_prepared[imgindex][pixindex] = k[0]
            pixindex += 1
    imgindex += 1
    pixindex = 0

clf = neighbors.KNeighborsClassifier(n_neighbors=1,algorithm='kd_tree')
clf.fit(x_tr_prepared, y_tr)
y_pred = clf.predict(x_val_prepared)

image_id = np.linspace(1,len(y_pred),num=len(y_pred)).astype(int)
header = ['Id','Class']
data = np.column_stack((image_id,y_pred))

with open('predictions1000n.csv','w',encoding='UTF8',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)