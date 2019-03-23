from sklearn.svm import SVC
from sklearn import tree

#height, weight and shoe size
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43], [168, 75, 41], [168, 77, 41]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male', 'female', 'female']

testData = [[190, 70, 43],[154, 75, 42],[181,65,40]]
testLabels = ['male','male','male']

sClassifier = SVC()
sClassifier.fit(X,Y)
sPrediction = sClassifier.predict(testData)
print sPrediction


dtcClasssifier = tree.DecisionTreeClassifier()
dtcClasssifier = dtcClasssifier.fit(X,Y)
dtcPrediction = dtcClasssifier.predict(testData)
print dtcPrediction