# Applying Nearest neighbors model with different number of neighbors and
# plotting the Accuracy for each model.

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import time

start = time.time()

X = pd.read_csv('noScalingPPTrainValues.csv', index_col='building_id')
y = pd.read_csv('train_labels.csv', index_col='building_id')

trainX, testX, trainy, testy = train_test_split(X, y, stratify=y,
                                                random_state=12)

numNeighbors = [i for i in range(5, 31)]
testAccuracies = []
trainAccuracies = []

for k in numNeighbors:
    clf = KNeighborsClassifier(n_neighbors=k, p=1, weights='distance',
                               n_jobs=8)
    clf.fit(trainX, trainy.values.ravel())
    yPred = clf.predict(testX)
    tyPred = clf.predict(trainX)
    testAccuracies.append(f1_score(testy, yPred, average='micro'))
    trainAccuracies.append(f1_score(trainy, tyPred, average='micro'))
    cmTest = confusion_matrix(testy, yPred)
    cmTrain = confusion_matrix(trainy, tyPred)
    cmTestNor = cmTest.astype('float') / cmTest.sum(axis=1)[:, np.newaxis]
    cmTrainNor = cmTrain.astype('float') / cmTrain.sum(axis=1)[:, np.newaxis]
    print('Normalized test Confusion Matrix for k = {} : \n'.format(k),
          cmTestNor, '\n')
    print('Normalized train Confusion Matrix for k = {} : \n'.format(k),
          cmTrainNor, '\n----------------------------------------------------\
    \n')

print('time : ' + str((time.time() - start)/60))

plt.subplot(2, 1, 1)
plt.plot(numNeighbors, testAccuracies, color='b', linestyle='-', marker='.',
         markersize=8)
plt.gca().set(xlabel='Number of neighbors', ylabel='Accuracy',
              title="Test accuracy of each model (weights='distance', p=1)",
              xticks=numNeighbors, xlim=(min(numNeighbors), max(numNeighbors)))
plt.grid(linestyle='--')
plt.subplot(2, 1, 2)
plt.plot(numNeighbors, trainAccuracies, color='r', linestyle='-', marker='.',
         markersize=8)
plt.gca().set(xlabel='Number of neighbors', ylabel='Accuracy',
              title="Train accuracy of each model (weights='distance', p=1)",
              xticks=numNeighbors, xlim=(min(numNeighbors), max(numNeighbors)))
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.6f}'))
plt.grid(linestyle='--')
plt.tight_layout()
plt.show()
