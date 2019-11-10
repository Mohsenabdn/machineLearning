# Applying Random Forest meta estimator using GridSearchCV. Besides, One can
# uncomment parts 1********** to 4********** and the plot section to see the
# accuracy for different hyperparameters.

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


X = pd.read_csv('noScalingPPTrainValues.csv', index_col='building_id')
y = pd.read_csv('train_labels.csv', index_col='building_id')

trainX, testX, trainy, testy = train_test_split(X, y, stratify=y,
                                                random_state=12)

# Make a comment this line

params = {'n_estimators': range(300, 700, 20), 'max_depth': range(20, 45),
          'min_samples_split': range(2, 13), 'max_features': range(8, 50, 2)}


# # 1**********
# # Setting max_depth, min_samples_split, and max_features to default values,
# # this peice of code finds accuracy for different n_estimators values.
#
# acc1 = []
# nEstimatorsList = [i for i in range(300, 701, 20)]
# for i in nEstimatorsList:
#     rfc1 = RandomForestClassifier(n_estimators=i, random_state=12, n_jobs=8)
#     rfc1.fit(trainX, trainy.values.ravel())
#     predy = rfc1.predict(testX)
#     acc1.append(f1_score(testy, predy, average='micro'))
#
#
# # 2**********
# # Setting n_estimators=520 and both min_samples_split and max_features to
# # default values, this peice of code finds accuracy for different max_depth
# # values.
#
# acc2 = []
# maxDepthList = [i for i in range(20, 45)]
# for i in maxDepthList:
#     rfc2 = RandomForestClassifier(n_estimators=520, max_depth=i,
#                                   random_state=12, n_jobs=8)
#     rfc2.fit(trainX, trainy.values.ravel())
#     predy = rfc2.predict(testX)
#     acc2.append(f1_score(testy, predy, average='micro'))
#
#
# # 3**********
# # Setting n_estimators=520, max_depth=33 and max_features to default value,
# # this peice of code finds accuracy for different min_samples_split values.
#
# acc3 = []
# minSampSplitList = [i for i in range(2, 13)]
# for i in minSampSplitList:
#     rfc3 = RandomForestClassifier(n_estimators=520, max_depth=33,
#                                   min_samples_split=i, random_state=12,
#                                   n_jobs=8)
#     rfc3.fit(trainX, trainy.values.ravel())
#     predy = rfc3.predict(testX)
#     acc3.append(f1_score(testy, predy, average='micro'))
#
#
# # 4**********
# # Setting n_estimators=520, max_depth=33, and min_samples_split=7, this peice
# # of code finds accuracy for different max_features values.
#
# acc4 = []
# maxFeaturesList = [i for i in range(8, 50, 2)]
# for i in maxFeaturesList:
#     rfc4 = RandomForestClassifier(n_estimators=520, max_depth=33,
#                                   min_samples_split=7, max_features=i,
#                                   random_state=12, n_jobs=8)
#     rfc4.fit(trainX, trainy.values.ravel())
#     predy = rfc4.predict(testX)
#     acc4.append(f1_score(testy, predy, average='micro'))


# Make comment this piece of code when running part 1********** to 4**********.

rfc = RandomForestClassifier(random_state=12, n_jobs=8)
gSearcher = GridSearchCV(rfc, param_grid=params, scoring='f1_micro', cv=5)
gSearcher.fit(trainX, trainy.values.ravel())
print('Best hyperparameters : \n', gSearcher.best_params_)
print('Best score : ', gSearcher.best_score_)
predy = gSearcher.predict(testX)
print('Test score : ', f1_score(testy, predy, average='micro'))


# # Plotting the accuracy of the models created in part 1**********.
#
# plt.figure(1)
# plt.plot(nEstimatorsList, acc1, linestyle='-', marker='.', markersize=8)
# plt.gca().set(xlabel='Number of estimators', ylabel='Accuracy',
#               title="Accuracy of each model (max_depth=None,\
#  min_samples_split=2, max_features='auto')", xticks=nEstimatorsList,
#               xlim=(min(nEstimatorsList), max(nEstimatorsList)))
# plt.grid(linestyle='--')
#
#
# # Plotting the accuracy of the models created in part 2**********.
#
# plt.figure(2)
# plt.plot(maxDepthList, acc2, linestyle='-', marker='.', markersize=8,
#          color='r')
# plt.gca().set(xlabel='Maximum depth', ylabel='Accuracy',
#               title="Accuracy of each model (n_estimators=520,\
#  min_samples_split=2, max_features='auto')", xticks=maxDepthList,
#               xlim=(min(maxDepthList), max(maxDepthList)))
# plt.grid(linestyle='--')
#
#
# # Plotting the accuracy of the models created in part 3**********.
#
# plt.figure(3)
# plt.plot(minSampSplitList, acc3, linestyle='-', marker='.', markersize=8)
# plt.gca().set(xlabel='Minimum samples split', ylabel='Accuracy',
#               title="Accuracy of each model (n_estimators=520, max_depth=33,\
#  max_features='auto')", xticks=minSampSplitList, xlim=(min(minSampSplitList),
#                                                        max(minSampSplitList)))
# plt.grid(linestyle='--')
#
#
# # Plotting the accuracy of the models created in part 4**********.
#
# plt.figure(4)
# plt.plot(maxFeaturesList, acc4, linestyle='-', marker='.', markersize=8,
#          color='r')
# plt.gca().set(xlabel='Maximum features', ylabel='Accuracy',
#               title="Accuracy of each model (n_estimators=520, max_depth=33,\
#  min_samples_split=7)", xticks=maxFeaturesList, xlim=(min(maxFeaturesList),
#                                                       max(maxFeaturesList)))
# plt.grid(linestyle='--')
#
# # Uncomment the following line when you want to see a plot or several plots.
#
# plt.show()
