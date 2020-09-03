#!/usr/bin/env python
from alibi.explainers import CounterFactual
from alibi.explainers import CounterFactualProto
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import os
import csv
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
import tensorflow as tf
import math
from scipy.stats import pearsonr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Ignores CPU Warning

# Reads in data
df = pd.read_csv(r'Data/data.csv')
data = np.array(df)
methodTitle = ["Logistic regression", "Discriminant analysis", \
"Decision tree", "Random forest", "Gradient boosting"]

# ------------------ Classes -----------------------------

# General class for a model classifier
class Classifier:

    global data

    def __init__(self, testSize, features, correlated, couple):
        # testSize is the proportion of data to be used as test data
        # features refers to the number of features
        # correlated and couple are used for the correlation tests, which are
        # activated if correlated=True and the couple is the indices of features to use

        self.x=[] # Inputs
        self.y = [] # Response
        self.features = features

        # Retrieves relevant features and only keeps non-null observations
        if (features == 2) and (correlated == False):
            featNum = [8, 12]
            featCat = []

        elif (features == 2) and (correlated == True):
            # Specifically for correlation tests
            featNum = [couple[0], couple[1]]
            featCat = []

        elif (features == 4):
            featNum = [7, 8, 12] # Indices of numbers features
            featCat = [4] # Categorical features

        elif (features == 5):
            featNum = [1, 7, 8, 12]
            featCat = [4]

        elif (features == 6):
            featNum = [1, 7, 8, 11, 12]
            featCat = [4]

        elif (features == 7):
            featNum = [1, 7, 8, 11, 12]
            featCat = [4, 5]

        elif (features == 8):
            featNum = [1, 6, 7, 8, 11, 12]
            featCat = [4, 5]

        elif (features == 9):
            featNum = [1, 3, 6, 7, 8, 11, 12]
            featCat = [4, 5]

        else:
          # 10 features
          featNum = [1, 2, 3, 6, 7, 8, 11, 12] # Indices of numbers features
          featCat = [4, 5] # Categorical features


        # Determines whether there are categorical variables to be encoded
        if (self.features==2):
          self.Encode = False
        else:
          self.Encode = True

        self.numEmpty = 0 # Number of rows not used
        self.catFeat = len(featCat) # Number of categorical features

        # Remove data if it has missing values
        num = []
        cat = []
        for row in data:
          flag = True
          numList = []
          catList = []

          # Checks for missing numbers
          for i in range(len(featNum)):
            numList.append(row[featNum[i]])
            if np.isnan(row[featNum[i]]):
              flag = False

          # Checks for missing categorical variables
          for i in range(len(featCat)):
            catList.append(row[featCat[i]])
            if str(row[featCat[i]]) == str(np.nan):
              flag = False

          # Checks for missing response variables
          if np.isnan(row[0]):
            flag = False

          if flag:  # Use row
            num.append(numList)
            if featCat:
              cat.append(catList)
            self.y.append(row[0])
          else: # Do not use row
            self.numEmpty = self.numEmpty+1

        self.lenNum = len(num[0])

        if cat:
          for i in range(len(cat)):
              for val in cat[i]:
                num[i].append(val)

        # Encodes all variables
        if cat:
          catVariables = []
          j = self.lenNum
          for i in range(len(featCat)):
            catVariables.append(j)
            j=j+1

          self.catVariables = catVariables
          colIndex = []
          for i in range(len(num[0])):
            colIndex.append(i)
          self.colIndex = colIndex[:]

          # Tells scikit learn model to handle categorical variables
          self.colTrans = make_column_transformer(
            (OneHotEncoder(handle_unknown='ignore'), catVariables), remainder='passthrough'
          )

        self.cat = cat
        self.x = num

        # Split data into training and test data
        xTrain, xTest, yTrain, yTest = train_test_split(self.x, self.y, test_size=testSize, shuffle=True)

        # Use one hot encoding
        if cat:
          self.enc = OneHotEncoder(handle_unknown='ignore')
          self.enc.fit(cat)
          cat = self.enc.transform(cat).toarray() # Encodes data
          self.categories = self.enc.categories_

          xTrain = pd.DataFrame(xTrain, columns=self.colIndex)
          xTrain = self.colTrans.fit_transform(xTrain)

          var = len(colIndex)
          for i in range(len(xTrain[0])-var):
            colIndex.append(len(colIndex))
          self.newColIndex = colIndex

          xTrain = pd.DataFrame(xTrain, columns=self.newColIndex)
          xTest =  pd.DataFrame(xTest, columns=self.colIndex)
          xTest = self.colTrans.transform(xTest)
          xTest = pd.DataFrame(xTest, columns=self.newColIndex)

          self.x =  pd.DataFrame(self.x, columns=self.colIndex)
          self.x = self.colTrans.transform(self.x)
          self.x = pd.DataFrame(self.x, columns=self.newColIndex)
          self.lenCat = len(cat[0])

        self.xTrain = xTrain
        self.xTest = xTest
        self.yTrain = yTrain
        self.yTest = yTest



    # Calculates accuracy (percentage of correct predictions)
    def accuracy(self):
        trainAcc = self.model.score(self.xTrain, self.yTrain) # Training data accuracy
        testAcc = self.model.score(self.xTest, self.yTest) # Test data accuracy
        self.trainAcc = trainAcc
        self.testAcc = testAcc

    # Shows number of data instances
    def instances(self):
        print("Data instances: {}".format(len(self.x)))

    # Generates classification report and saves relevant statistics
    def generateReport(self):
        self.report = metrics.classification_report(self.yTest, self.model.predict(self.xTest))
        reportList = self.report.split()
        self.precision0 = reportList[5]
        self.recall0 = reportList[6]
        self.fScore0 = reportList[7]
        self.support0 = reportList[8]
        self.precision1 = reportList[10]
        self.recall1 = reportList[11]
        self.fScore1 = reportList[12]
        self.support1 = reportList[13]

    # Shows classification report
    def showReport(self):
        self.generateReport()
        print("\nClassification report:\n{}".format(self.report))

    # Returns model accuracy and classification report statistics
    def statistics(self):
        self.accuracy()
        self.generateReport()
        return self.trainAcc, self.testAcc, self.precision0, self.recall0, \
        self.precision1, self.recall1

    # Calculates area under the ROC curve (rocScore)
    def rocScore(self):
        prob = self.model.predict_proba(self.xTest)
        self.posProb = [] # probability of positive class
        # only takes the probabilities for class 1 (poor credit score)
        for item in prob:
            self.posProb.append(item[1])
        self.rocScore = metrics.roc_auc_score(self.yTest, self.posProb)


    # Returns relevant ROC values
    def rocValues(self):
        self.rocScore()
        fpr, tpr, thresholds = metrics.roc_curve(self.yTest, self.posProb)
        return fpr, tpr, self.rocScore

    # Returns training and test data
    def getData(self):
        return self.xTrain, self.yTrain, self.xTest, self.yTest

    # Returns uncoded training and test data
    def getUncodedData(self):
        uTrain = []
        xDummyTrain = []
        for i in self.xTrain:
          arr = []
          for item in i:
            arr.append(item)
          xDummyTrain.append(arr)
        for i in xDummyTrain:
          Xarray = self.decode(i)
          lst = []
          for j in Xarray:
            lst.append(j)
          uTrain.append(lst)

        uTest = []
        xDummyTest = []
        for i in self.xTest:
          emptyArray = []
          for item in i:
            emptyArray.append(item)
          xDummyTest.append(emptyArray)
        for i in xDummyTest:
          Xarray = self.decode(i)
          lst = []
          for j in Xarray:
            lst.append(j)
          uTest.append(lst)

        return uTrain, self.yTrain, uTest, self.yTest

    # Returns predictions for training and test data
    def getPredictions(self):
        predictTrain = self.model.predict(self.xTrain)
        predictTest = self.model.predict(self.xTest)
        return predictTrain, predictTest

    # Returns predicted probabilities
    def predict(self, x):
        return self.model.predict_proba(x)

    # Shows how much of data is used
    def missingValues(self):
        print("Rows with missing values: {}".format(self.numEmpty))
        print("Useful rows: {}".format(len(self.x)))

    # Returns number of features
    def numFeatures(self):
        return len(self.x[0])

    # Number of features before categorical data is encoded
    def numFeaturesBeforeEnc(self):
      return self.features

    # Number of categorical features
    def numCatFeatures(self):
      return self.catFeat

    # Encodes single data point
    def encode(self, val):
        # Data instances are given with categorical features last
        if self.Encode:
          numPart = val[:len(val)-self.catFeat]
          catPart = val[len(val)-self.catFeat:]
          encodedPart = self.enc.transform([catPart]).toarray()

          lst = []
          for i in range(len(encodedPart[0])):
            lst.append(encodedPart[0][i])
          for i in range(len(numPart)):
            lst.append(numPart[i])
          return lst

        else:
          return val

    # Decodes single data point
    def decode(self, val):
        if self.Encode:
          # Length of categorical features when encoded
          encodedPart = val[:self.lenCat]
          numPart = val[self.lenCat:]
          catPart = self.enc.inverse_transform([encodedPart])
          #newVal = np.append(numPart, catPart)
          newVal = np.append(numPart, catPart)
          return newVal
        else:
          return val

    # Returns dictionary of categories for counterfactual prototype function
    def categoriesDict(self):
        dictCats = {} # dictionary of categories
        j = 0 # length of numerical features
        for i in range(len(self.categories)):
            value = len(self.categories[i]) # number of categories
            key = j # column where category starts (starting from index 0)
            j=j+value
            dictCats[key] = value
        return dictCats

    # Find correlation between two variables - only use if features is 2
    def correlation(self):
        var1 = []
        var2 = []
        for i in self.x:
          var1.append(i[0])
          var2.append(i[1])
        coef, dummy = pearsonr(var1, var2)
        return coef



# Logistic regression
class LR(Classifier):

    def __init__(self, features, testSize, correlated, couple):
        super().__init__(testSize, features, correlated, couple)
        model = LogisticRegression(solver='liblinear', random_state=0)
        if self.cat:
          model = make_pipeline(self.colTrans, model)

        model.fit(self.xTrain, self.yTrain)

        if self.cat:
          self.x = self.x.values.tolist()
          self.xTrain = self.xTrain.values.tolist()
          self.xTest = self.xTest.values.tolist()

        self.model = model

# Discriminant analysis
class DA(Classifier):

    def __init__(self, features, testSize, correlated, couple, type):
        super().__init__(testSize, features, correlated, couple)
        if type == "LDA": # Linear discriminant analysis
            model = LinearDiscriminantAnalysis() # Uses svd and no shrinkage
        elif type=="QDA": # Quadratic discriminant analysis
            model = QuadraticDiscriminantAnalysis()
        else: # Regularised discriminant analysis
            model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            # Change shrinkage to between 0 and 1 or use 'auto' (determines optimal shrinkage)

        if self.cat:
          model = make_pipeline(self.colTrans, model)

        model.fit(self.xTrain, self.yTrain)

        if self.cat:
          self.x = self.x.values.tolist()
          self.xTrain = self.xTrain.values.tolist()
          self.xTest = self.xTest.values.tolist()

        self.model = model


# Abstract class for classifiers where feature importance can be found
class FeatImportance(Classifier):

    # Shows feature importances
    def importance(self):
        print("\nFeature Importances:")
        for i in range(len(self.x[0])):
            print("{}: {}".format(i+1,self.model.feature_importances_[i]))

    # Return feature importances
    def retrieveImportance(self):
        return self.model.feature_importances_


# Classification tree
class Tree(FeatImportance):

    def __init__(self, features, testSize, maxDepth):
        super().__init__(testSize, features)
        model = DecisionTreeClassifier(max_depth=maxDepth)
        model.fit(self.xTrain, self.yTrain)
        self.model = model


    # Shows tree depth
    def depth(self):
        print("\nTree depth: {}".format(self.model.get_depth()))

    # Shows number of leaves (terminal nodes)
    def leaves(self):
        print("Number of leaves: {}".format(self.model.get_n_leaves()))


# Random forest
class Forest(FeatImportance):
    def __init__(self, features, testSize, maxDepth, numTrees):
        super().__init__(testSize, features)
        model = RandomForestClassifier(n_estimators=numTrees, max_depth=maxDepth)
        model.fit(self.xTrain, self.yTrain)
        self.model = model

# Gradient Boosting Machine
class GBM(FeatImportance):

    def __init__(self, features, testSize, lossParameter):
        super().__init__(testSize, features)
        model = GradientBoostingClassifier(loss=lossParameter)
        model.fit(self.xTrain, self.yTrain)
        self.model = model

# Allows dictionary items to be accessed by object attributes
class dictObject(object):
    def __init__(self, obj):
        self.__dict__ = obj


# Simple method for generating counterfactual explanations
def CF(classifier, instance, output, CFtype, returnValue, numTrainParam):
  """
  classifier is the model to generate explanations for, instance is the instance
  to be explained, output is whether information should be displayed (True or False)
  returnValue determines which data is returned
  The number of close data points to the counterfactual, within a Gower distance
  of numTrainParam are found
  """
  tf.keras.backend.clear_session()
  tf.compat.v1.disable_eager_execution()
  predict_fn = lambda x: classifier.predict(x) # model predicted probabilies
  xTrain, yTrain, xTest, yTest = classifier.getData()
  xTrain = np.array(xTrain)
  xTest= np.array(xTest)

  # Encode instance to be explained and reshape
  instance = classifier.encode(instance)
  obj = np.array(instance).reshape(1,classifier.numFeatures())

  # Finds range of training data features (feature-wise limits)
  featBeforeEnc = classifier.numFeaturesBeforeEnc() # No. of features before being encoded
  catFeat = classifier.numCatFeatures() # No. of categorical features
  minValue = []
  maxValue = []
  for i in range(classifier.numFeatures()):
    input = [] # considers one feature at at a time
    for item in xTrain:
      input.append(item[i])
    minValue.append(min(input)) # min of feature
    maxValue.append(max(input)) # max of feature
  featRange = (tuple(minValue), tuple(maxValue)) # Feature-wise limits
  #featRange = (min(xTrain), max(xTrain)) # Global option

  shape = (1,) + xTrain.shape[1:]

  # Initialise counterfactual object
  cf = CounterFactual(predict_fn, shape, target_proba=0.75,
                    tol=0.25, feature_range=featRange,
                    lam_init=0.01, early_stop=100)

  # Time it takes to generate explanations
  start_time = time()
  explanation = cf.explain(obj)
  timeTaken = time() - start_time

  # Extract results about original instance
  e = dictObject(explanation.cf)
  origProb = explanation.orig_proba
  origClass = explanation.orig_class
  if origClass == 0:
      probs = [origProb, 1-origProb]
  else:
      probs = [1-origProb, origProb]
  origInstance = classifier.decode(obj[0])

  # Finds counterfactual with smallest distance from original instance (l1 norm)
  input = e.X[0]
  prob = e.proba[0]
  newClass = explanation.cf['class']
  dist1 = e.distance
  minFeat = 0 # Calculates number of features

  counter = classifier.decode(input) # current counterfactual
  for i in range(len(origInstance)):
    if origInstance[i] != counter[i]:
      minFeat = minFeat+1

  # Finds counterfactual with smallest number of features changed
  allCEs = explanation.all[0]  # Info about all counterfactuals
  numCF = len(allCEs) # Number of counterfactuals
  vecAllCEs = [] # Vector storing all counterfactual explanations

  for i in range(len(allCEs)):
        cfactual = dictObject(allCEs[i])
        vecAllCEs.append(cfactual.X[0])
        feat = 0
        counter = classifier.decode(cfactual.X[0]) # current counterfactual
        if (CFtype == "features"):
          for i in range(len(origInstance)):
            if origInstance[i] != counter[i]:
              feat = feat+1
          if feat < minFeat:
                minFeat = feat
                input = cfactual.X[0]
                prob = cfactual.proba[0]
                newClass = allCEs[i]['class']


  # Calculate Gower distance
  def findDistance(instance, counterfactual):
    dist = 0
    for i in range(featBeforeEnc-catFeat):
      rangeVal = maxValue[i]-minValue[i]
      dist = dist + math.fabs(instance[i]-counterfactual[i])/rangeVal
    for i in range(catFeat):
      if instance[featBeforeEnc-1-i] == counterfactual[featBeforeEnc-1-i]:
        indicate = 0 # indicator function
      else:
        indicate = 1
      dist = dist + indicate
    dist = dist/featBeforeEnc
    return dist

  # Find diversity estimate
  numCF = len(vecAllCEs)
  diversity = 0
  if len(vecAllCEs) >1:
    for cf in range(len(vecAllCEs)):
      sum = 0
      for j in range(len(vecAllCEs)):
        if j!=cf:
          distBetweenCFs = findDistance(vecAllCEs[cf], vecAllCEs[j])
          sum = sum + distBetweenCFs
      sum = sum/(len(vecAllCEs)-1)
      diversity = diversity+sum

    diversity= diversity/len(vecAllCEs) # Average distance between counterfactuals


  # Distance of final counterfactual found from original instance
  cInstance = classifier.decode(input)
  dist = findDistance(origInstance, cInstance)


  # Find which features have changed
  featChanged = []
  for i in range(len(origInstance)):
    if origInstance[i]==cInstance[i]:
      featChanged.append(0)
    else:
      featChanged.append(1)

  # Find closeness of final counterfactual to nearest training point
  nearestTrain = findDistance(classifier.decode(xTrain[0]), cInstance)
  closeI = 0
  for i in range(len(xTrain)):
    closeVal = findDistance(classifier.decode(xTrain[i]), cInstance)
    if closeVal < nearestTrain:
      nearestTrain = closeVal
      closeI = i

  # Find average closeness of final counterfactual to nearest two training points
  close2 = findDistance(classifier.decode(xTrain[0]), cInstance)
  for i in range(len(xTrain)):
    closeVal = findDistance(classifier.decode(xTrain[i]), cInstance)
    if closeI != i:
      if closeVal < close2:
        close2 = closeVal

  nearestTrain2 = (nearestTrain + close2)/2

  # Find number of training points that are a specific distance to the counterfactual
  numTraining = 0
  for i in range(len(xTrain)):
    closeVal = findDistance(classifier.decode(xTrain[i]), cInstance)
    if closeVal <= numTrainParam:
      numTraining = numTraining + 1

  if output: # displays output
      print("\nOriginal instance: {}".format(origInstance))
      print("Original probabilities: {}".format(probs))
      print("Original prediction: {}".format(origClass))
      print("\nCounterfactual instance: {}".format(cInstance))
      print("Counterfactual probabilities: {}".format(prob))
      print("Counterfactual prediction: {}".format(newClass))
      print("\nGower distance from original instance: {}".format(dist))
      print("Diversity of counterfactuals: {}".format(diversity))
      print("Number of counterfactuals generated: {}".format(numCF))
      print("Number of features changed: {}".format(minFeat))
      print('Explanation time {:.3f} sec'.format(timeTaken))
      print('\nDistance of counterfactual to nearest training instance: {}'.format(nearestTrain))
      print('Average Distance of counterfactual to nearest two training instances: {}'.format(nearestTrain2))
      print('Number of training points within distance {} of counterfactual: {}'.format(numTrainParam, numTraining))

  if returnValue == "data":
    # returns data about counterfactuals found
    return minFeat, dist, timeTaken, numCF, diversity, nearestTrain, numTraining, featChanged

  if returnValue == "CF":
    # returns counterfactual value and its class
    return cInstance, newClass



# Prototype method for generating counterfactual explanations
def protoCF(classifier, instance, output, CFtype, returnValue, numTrainParam):
  """
  classifier is the model to generate explanations for, instance is the instance
  to be explained, output is whether information should be displayed (True or False)
  returnValue determines which data is returned
  The number of close data points to the counterfactual, within a Gower distance
  of numTrainParam are found
  """

  tf.keras.backend.clear_session()
  tf.compat.v1.disable_eager_execution()
  predict_fn = lambda x: classifier.predict(x) # model predicted probabilies
  xTrain, yTrain, xTest, yTest = classifier.getData()
  xTrain = np.array(xTrain)
  xTest= np.array(xTest)

  # Encode instance to be explained and reshape
  instance = classifier.encode(instance)
  obj = np.array(instance).reshape(1,classifier.numFeatures())

  # Finds range of training data features (feature-wise limits)
  featBeforeEnc = classifier.numFeaturesBeforeEnc() # No. of features before being encoded
  catFeat = classifier.numCatFeatures() # No. of categorical features
  minValue = []
  maxValue = []
  for i in range(featBeforeEnc-catFeat):
    input = [] # considers one numerical feature at a time
    for item in xTrain:
      input.append(item[i])
    minValue.append(min(input)) # minumim value of feature
    maxValue.append(max(input)) # maximum value of feature
  for i in range(catFeat): # Dummy values for categorical features
    minValue.append(-10e2)
    maxValue.append(10e2)
  minValue2 = np.array(minValue).reshape((1,featBeforeEnc))
  maxValue2 = np.array(maxValue).reshape((1,featBeforeEnc))
  featRange = (minValue2, maxValue2)

  shape = (1,) + xTrain.shape[1:]

  # Initialise counterfactual object
  if classifier.Encode:
    catVars = classifier.categoriesDict() # dictionary for location of categorical variables
    cf = CounterFactualProto(predict_fn, shape, use_kdtree=True, cat_vars = catVars, ohe=True, feature_range=featRange)
  else:
    cf = CounterFactualProto(predict_fn, shape, use_kdtree=True, feature_range=featRange)

  cf.fit(xTrain, update_feature_range=True)

  # Time it takes to generate explanations
  start_time = time()
  explanation = cf.explain(obj)
  timeTaken = time() - start_time


  # Extract results about original instance
  e = dictObject(explanation.cf)
  origProb = explanation.orig_proba[0][0]
  origClass = explanation.orig_class
  if origClass == 0:
      probs = [origProb, 1-origProb]
  else:
      probs = [1-origProb, origProb]
  origInstance = classifier.decode(obj[0])

  # Finds counterfactual with smallest distance from original instance
  input = e.X[0]
  probCF = e.proba[0]
  newClass = explanation.cf['class']

  # Calculates number of features
  minFeat = 0
  counter = classifier.decode(input) # current counterfactual
  for i in range(len(origInstance)):
    if origInstance[i] != counter[i]:
      minFeat = minFeat+1

  allCEs = explanation.all # Info about all counterfactuals
  vecAllCEs = [] # Vector storing all counterfactual explanations
  numCF = 0 # Number of counterfactuals

  for i in range(len(allCEs)):
    CFlst = allCEs[i]
    for j in range(len(CFlst)):
      numCF = numCF +1
      cfactual = CFlst[j][0]
      vecAllCEs.append(cfactual)
      feat = 0
      counter = classifier.decode(cfactual) # current counterfactual
      if (CFtype == "features"):
        for i in range(len(origInstance)):
            if origInstance[i] != counter[i]:
              feat = feat+1
        if feat < minFeat:
          minFeat = feat
          input = cfactual
          probCF = classifier.predict([cfactual])[0]

          if probCF[0]>0.5:
            newClass = 0
          else:
            newClass = 1

  # Calculate Gower distance
  def findDistance(instance, counterfactual):
    dist = 0
    for i in range(featBeforeEnc-catFeat):
      rangeVal = maxValue[i]-minValue[i]
      dist = dist + math.fabs(instance[i]-counterfactual[i])/rangeVal
      numDist = dist/featBeforeEnc # distance of just numerical values
    for i in range(catFeat):
      if instance[featBeforeEnc-1-i] == counterfactual[featBeforeEnc-1-i]:
        indicate = 0 # indicator function
      else:
        indicate = 1
      dist = dist + indicate
    dist = dist/featBeforeEnc
    return dist

  # Find diversity estimate
  numCF = len(vecAllCEs)
  diversity = 0
  if len(vecAllCEs) >1:
    for cf in range(len(vecAllCEs)):
      sum = 0
      for j in range(len(vecAllCEs)):
        if j!=cf:
          distBetweenCFs = findDistance(vecAllCEs[cf], vecAllCEs[j])
          sum = sum + distBetweenCFs
      sum = sum/(len(vecAllCEs)-1)
      diversity = diversity+sum

    diversity= diversity/len(vecAllCEs) # Average distance between counterfactuals

  # Distance of final counterfactual found from original instance
  cInstance = classifier.decode(input)
  dist = findDistance(origInstance, cInstance)

  # Find which features have changed
  featChanged = []
  for i in range(len(origInstance)):
    if origInstance[i]==cInstance[i]:
      featChanged.append(0)
    else:
      featChanged.append(1)

  # Find closeness of final counterfactual to nearest training point
  nearestTrain = findDistance(classifier.decode(xTrain[0]), cInstance)
  closeI = 0
  for i in range(len(xTrain)):
    closeVal = findDistance(classifier.decode(xTrain[i]), cInstance)
    if closeVal < nearestTrain:
      nearestTrain = closeVal
      closeI = i

  # Find average closeness of final counterfactual to nearest two training points
  close2 = findDistance(classifier.decode(xTrain[0]), cInstance)
  for i in range(len(xTrain)):
    closeVal = findDistance(classifier.decode(xTrain[i]), cInstance)
    if closeI != i:
      if closeVal < close2:
        close2 = closeVal

  nearestTrain2 = (nearestTrain + close2)/2

  # Find number of training points that are a specific distance to the counterfactual
  numTraining = 0
  for i in range(len(xTrain)):
    closeVal = findDistance(classifier.decode(xTrain[i]), cInstance)
    if closeVal <= numTrainParam:
      numTraining = numTraining + 1

  if output: # displays output
      print("\nOriginal instance: {}".format(origInstance))
      print("Original probabilities: {}".format(probs))
      print("Original prediction: {}".format(origClass))
      print("\nCounterfactual instance: {}".format(cInstance))
      print("Counterfactual probabilities: {}".format(probCF))
      print("Counterfactual precition: {}".format(newClass))
      print("\nGower Distance from original instance: {}".format(dist))
      print("Diversity of counterfactuals: {}".format(diversity))
      print("Number of counterfactuals generated: {}".format(numCF))
      print("Number of features changed: {}".format(minFeat))
      print('Explanation time {:.3f} sec'.format(timeTaken))
      print('\nDistance of counterfactual to nearest training instance: {}'.format(nearestTrain))
      print('Average Distance of counterfactual to nearest two training instances: {}'.format(nearestTrain2))
      print('Number of training points within distance {} of counterfactual: {}'.format(numTrainParam, numTraining))

  if returnValue == "data":
    # returns data about counterfactuals found
    return minFeat, dist, timeTaken, numCF, diversity, nearestTrain, numTraining, featChanged

  if returnValue == "CF":
    # returns counterfactual value and its class
    return cInstance, newClass


# Saves data of features changed and distance
def generateData(classifier, typeCF, cFactuals, proto, numTrainParam):
    # if proto is true, use counterfactuals guided by prototypes
    xTrain, yTrain, xTest, yTest = classifier.getUncodedData()
    # Counterfactuals are generated for xTest data
    numCEs = cFactuals
    featureVal = []
    distance = []
    timeTaken = []
    numCFVal = []
    divVal = []
    nearestTrain = []
    numTrain = []
    featC = []

    for i in range(numCEs):
        if proto:
            fVal, dist, time, numCF, diversity, nearT, numT, featChanged  = protoCF(classifier, xTest[i], False, typeCF, "data", numTrainParam)
        else:
            fVal, dist, time, numCF, diversity, nearT, numT, featChanged = CF(classifier, xTest[i], False, typeCF, "data",numTrainParam)
        featureVal.append(fVal)
        distance.append(dist)
        timeTaken.append(time)
        numCFVal.append(numCF)
        divVal.append(diversity)
        nearestTrain.append(nearT)
        numTrain.append(numT)
        featC.append(featChanged)

    # Writes data to a file
    if proto==True:
      fileName = "featDistProto.csv"
    else:
      fileName = "featDistCF.csv"
    with open(fileName, "w") as csvfile:
      cols = ['features', 'distance', 'time', 'numCF', 'diversity', \
              'nearestTrain', 'numTrain', 'featChanged']
      writeToFile = csv.DictWriter(csvfile, fieldnames=cols)
      writeToFile.writeheader()
      for i in range(len(featureVal)):
        writeToFile.writerow({'features':featureVal[i], 'distance':distance[i], \
                         'time':timeTaken[i], 'numCF':numCFVal[i], \
                         'diversity':divVal[i], 'nearestTrain':nearestTrain[i], \
                         'numTrain':numTrain[i], 'featChanged':featC[i]})



# Creates CDF of features and distance
def CDF(radius):

    # Retrieves data of counterfactuals
    cols = ['features', 'distance', 'time', 'numCF', 'diversity', \
              'nearestTrain', 'numTrain', 'featChanged']
    df = pd.read_csv("featDistCF.csv", usecols=cols)
    featureVal = list(df["features"])
    distance = list(df['distance'])
    diversity = list(df['diversity'])
    df2 = pd.read_csv("featDistProto.csv", usecols=cols)
    featureVal2 = list(df2["features"])
    distance2 = list(df2['distance'])
    diversity2 = list(df2['diversity'])
    nearT = list(df['nearestTrain'])
    nearT2 = list(df2['nearestTrain'])
    numT =list(df['numTrain'])
    numT2 = list(df2['numTrain'])

    # Calculate mean, median and standard deviation
    meanF = np.mean(featureVal)
    meanD = np.mean(distance)
    meanDiv = np.mean(diversity)
    meanNearT = np.mean(nearT)
    meanNumT = np.mean(numT)
    sdF = np.std(featureVal)
    sdD = np.std(distance)
    sdDiv = np.std(diversity)
    sdNearT = np.std(nearT)
    sdNumT = np.std(numT)
    meanF2 = np.mean(featureVal2)
    meanD2 = np.mean(distance2)
    meanDiv2 = np.mean(diversity2)
    meanNearT2 = np.mean(nearT2)
    meanNumT2 = np.mean(numT2)
    sdF2 = np.std(featureVal2)
    sdD2 = np.std(distance2)
    sdDiv2 = np.std(diversity2)
    sdNearT2 = np.std(nearT2)
    sdNumT2 = np.std(numT2)

    # Distance CDF
    bins = 'auto'
    hist, binEdges = np.histogram(distance, bins=bins)
    hist2, binEdges2 = np.histogram(distance2, bins=bins)
    cdf = np.cumsum(hist)
    cdf2 = np.cumsum(hist2)
    plt.figure(0)
    plt.plot(binEdges[1:], cdf/cdf[-1])
    plt.plot(binEdges2[1:], cdf2/cdf2[-1])
    plt.xlabel("Distance (Gower)")
    plt.ylabel("Instances explained (%)")
    plt.title("CDF of distance between original instance and counterfactual")
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    simpleStr = "Simple \u03BC={:.3f}, \u03C3={:.3f}".format(meanD, sdD)
    protoStr = "Prototype \u03BC={:.3f}, \u03C3={:.3f}".format(meanD2, sdD2)
    plt.legend([simpleStr, protoStr], fontsize=8, loc="best")
    plt.savefig('dist.png')


    plt.figure()
    fontSize = 14
    bins='auto'
    xlabelSize = 12
    barWidth = 0.6
    bars, freq = np.unique(featureVal, return_counts=True)
    plt.bar(bars, freq, width = barWidth, color='#1B64BD',alpha=0.7)
    plt.xlabel("Number of features changed", fontsize=fontSize)
    plt.ylabel("Frequency", fontsize=fontSize)
    plt.title("Number of features changed when generating counterfactuals", fontsize=fontSize)
    plt.xticks(bars, fontsize=xlabelSize)
    plt.yticks()
    plt.savefig('feat.png')

    # Diversity CDF
    bins = 'auto'
    hist, binEdges = np.histogram(diversity, bins=bins)
    hist2, binEdges2 = np.histogram(diversity2, bins=bins)
    cdf = np.cumsum(hist)
    cdf2 = np.cumsum(hist2)
    plt.figure()
    plt.plot(binEdges[1:], cdf/cdf[-1])
    plt.plot(binEdges2[1:], cdf2/cdf2[-1])
    plt.xlabel("Diversity (Gower distance between counterfactuals)")
    plt.ylabel("Instances explained (%)")
    plt.title("CDF of average distance between counterfactuals")
    simpleStr = "Simple \u03BC={:.3f}, \u03C3={:.3f}".format(meanDiv, sdDiv)
    protoStr = "Prototype \u03BC={:.3f}, \u03C3={:.3f}".format(meanDiv2, sdDiv2)
    plt.legend([simpleStr, protoStr], fontsize=8, loc="best")
    plt.savefig('div.png')

    # CDF of distance to nearest training point
    bins = 'auto'
    hist, binEdges = np.histogram(nearT, bins=bins)
    hist2, binEdges2 = np.histogram(nearT2, bins=bins)
    cdf = np.cumsum(hist)
    cdf2 = np.cumsum(hist2)
    plt.figure()
    plt.plot(binEdges[1:], cdf/cdf[-1])
    plt.plot(binEdges2[1:], cdf2/cdf2[-1])
    plt.xlabel("Gower distance")
    plt.ylabel("Instances explained (%)")
    plt.title("CDF of distance from counterfactual to nearest data point")
    simpleStr = "Simple \u03BC={:.3f}, \u03C3={:.3f}".format(meanNearT, sdNearT)
    protoStr = "Prototype \u03BC={:.3f}, \u03C3={:.3f}".format(meanNearT2, sdNearT2)
    plt.legend([simpleStr, protoStr], fontsize=8, loc="best")
    plt.savefig('nearT.png')

    # CDF of number of data points within a distance of counterfactual
    bins = 'auto'
    hist, binEdges = np.histogram(numT, bins=bins)
    hist2, binEdges2 = np.histogram(numT2, bins=bins)
    cdf = np.cumsum(hist)
    cdf2 = np.cumsum(hist2)
    plt.figure()
    plt.plot(binEdges[1:], cdf/cdf[-1])
    plt.plot(binEdges2[1:], cdf2/cdf2[-1])
    plt.xlabel("Number of data points ")
    plt.ylabel("Instances explained (%)")
    plt.title("CDF of number of data points within a Gower distance {} of the counterfactual found".format(radius))
    simpleStr = "Simple \u03BC={:.3f}, \u03C3={:.3f}".format(meanNumT, sdNumT)
    protoStr = "Prototype \u03BC={:.3f}, \u03C3={:.3f}".format(meanNumT2, sdNumT2)
    plt.legend([simpleStr, protoStr], fontsize=8, loc="best")
    plt.savefig('numT.png')


# Average time taken to generate explanations
def averageTime():

    # Retrieves data of counterfactuals
    cols = ['features', 'distance', 'time', 'numCF', 'diversity', \
              'nearestTrain', 'numTrain', 'featChanged']
    df = pd.read_csv("featDistCF.csv", usecols=cols)
    df2 = pd.read_csv("featDistProto.csv", usecols=cols)
    time = list(df['time'])
    time2 = list(df2['time'])
    mean = np.mean(time)
    mean2 = np.mean(time2)

    print("Average time of simple method: {:.2f} seconds".format(mean))
    print("Average time of prototype method: {:.2f} seconds".format(mean2))


# Average number of counterfactuals generated
def numCfactuals():

    # Retrieves data of counterfactuals
    cols = ['features', 'distance', 'time', 'numCF', 'diversity', \
              'nearestTrain', 'numTrain', 'featChanged']
    df = pd.read_csv("featDistCF.csv", usecols=cols)
    df2 = pd.read_csv("featDistProto.csv", usecols=cols)
    cf = list(df['numCF'])
    cf2 = list(df2['numCF'])
    mean = np.mean(cf)
    mean2 = np.mean(cf2)

    print("Average no. counterfactuals of simple method: {}".format(mean))
    print("Average no. counterfactuals of prototype method: {}".format(mean2))


# Reliability of counterfactuals to training data
def reliability():

    # Retrieves data of counterfactuals
    cols = ['features', 'distance', 'time', 'numCF', 'diversity', \
              'nearestTrain', 'numTrain', 'featChanged']
    df = pd.read_csv("featDistCF.csv", usecols=cols)
    df2 = pd.read_csv("featDistProto.csv", usecols=cols)
    nearT = list(df['nearestTrain'])
    nearT2 = list(df2['nearestTrain'])
    numT =list(df['numTrain'])
    numT2 = list(df2['numTrain'])

    meanNearT = np.mean(nearT)
    meanNearT2 = np.mean(nearT2)
    meanNumT = np.mean(numT)
    meanNumT2 = np.mean(numT2)

    print('Simple method distance to nearest training instance: {}'.format(meanNearT))
    print('Prototype method distance to nearest training instance: {}'.format(meanNearT2))
    print('Simple method no. training points within distance 0.14 of counterfactual: {}'.format(meanNumT))
    print('Prototype method no. training points within distance 0.14 of counterfactual: {}'.format(meanNumT2))



# Find feature importance based on features most commonly changed when generating counterfactuals
def featureImportance():

    # Retrieves data of counterfactuals
    cols = ['features', 'distance', 'time', 'numCF', 'diversity', \
              'nearestTrain', 'numTrain', 'featChanged']
    df = pd.read_csv("featDistCF.csv", usecols=cols)
    df2 = pd.read_csv("featDistProto.csv", usecols=cols)
    fC = list(df['featChanged'])
    fC2 = list(df2['featChanged'])

    newData = []
    for item in fC:
      val = list(item)
      vec = []
      for i in range(len(val)):
        if (i+2) % 3 == 0:
          vec.append(int(val[i]))

      newData.append(vec)

    newData2 = []
    for item in fC2:
      val = list(item)
      vec = []
      for i in range(len(val)):
        if (i+2) % 3 == 0:
          vec.append(int(val[i]))

      newData2.append(vec)

    simpleSum = []
    for i in range(len(newData[0])): # Iterate through number of features
      sum = 0
      for j in newData: # Iterate through features changed for each counterfactual
        sum = sum + j[i]

      simpleSum.append(sum)

    protoSum = []
    for i in range(len(newData2[0])): # Iterate through number of features
      sum = 0
      for j in newData2: # Iterate through features changed for each counterfactual
        sum = sum + j[i]

      protoSum.append(sum)

    features = len(newData[0]) # number of features

    # Finding feature names
    if features == 2:
      names = ['Delinquent credit lines', 'Debt to income', ]
    elif features == 4:
      names = ['Derogatory reports', 'Delinquent credit lines', \
               'Debt to income', 'Reason for loan']
    elif features == 5:
      names = ['Loan amount requested','Derogatory reports', 'Delinquent credit lines', \
               'Debt to income', 'Reason for loan']
    elif features == 6:
      names = ['Loan amount requested','Derogatory reports', 'Delinquent credit lines', \
               'Credit lines','Debt to income', 'Reason for loan']
    elif features == 7:
      names = ['Loan amount requested','Derogatory reports', 'Delinquent credit lines', \
               'Credit lines','Debt to income', 'Reason for loan', 'Job']
    elif features == 8:
      names = ['Loan amount requested','Years in job', 'Derogatory reports', \
               'Delinquent credit lines', 'Credit lines','Debt to income', \
               'Reason for loan', 'Job']
    elif features == 9:
      names = ['Loan amount requested','Value of collateral''Years in job', \
               'Derogatory reports', 'Delinquent credit lines', 'Credit lines',\
               'Debt to income', 'Reason for loan', 'Job']

    # Figure parameters
    bins='auto'
    fontSize = 14
    xlabelSize = 12
    barWidth = 0.7
    rotation=25 # 0
    method = ['simple', 'prototype']
    bars = []
    for i in range(features):
      bars.append(i+1)
    category = np.arange(len(bars))

    # Simple method
    plt.figure(0, figsize=[10,8])
    plt.bar(category, simpleSum, width = barWidth, color='#1B64BD',alpha=0.7)
    plt.xlabel("Feature", fontsize=fontSize)
    plt.ylabel("Number of times feature is changed", fontsize=fontSize)
    plt.title("Feature importance for {} method".format(method[0]), fontsize=fontSize)
    plt.xticks(category, names, fontsize=xlabelSize, rotation=rotation)
    plt.yticks()
    plt.savefig('fImpS.png')

    # Prototype method
    plt.figure(1, figsize=[10,8])
    plt.bar(category, protoSum, width = barWidth, color='#1B64BD',alpha=0.7)
    plt.xlabel("Feature", fontsize=fontSize)
    plt.ylabel("Number of times feature is changed", fontsize=fontSize)
    plt.title("Feature importance for {} method".format(method[1]), fontsize=fontSize)
    plt.xticks(category, names, fontsize=xlabelSize, rotation=rotation)
    plt.yticks()
    plt.savefig('fImpP.png')

    # Compare with other feature importance measures (Tree, Forest, GBM etc)
    tr = Tree(features, 0.25, 3)
    #imp = tr.retrieveImportance()

    plt.show()


# Visual plot of predicted training data, instance to be explained and counterfactual
def decisionBoundary(classifier, instance, numTrainParam): # Only works for 2 features
  predictTrain, predictTest=classifier.getPredictions()
  xTrain, yTrain, xTest, yTest = classifier.getData()
  fontSize = 14
  plt.figure(0, figsize=[6,4])
  for i in range(len(yTrain)):
        if predictTrain[i]==0:
            # Good credit risk - blue
            plt.plot(xTrain[i][1],xTrain[i][0],'ob', markersize=2)
        else:
            # Bad credit risk - red
            plt.plot(xTrain[i][1],xTrain[i][0],'or', markersize=2)

  # Plot instance to be explained as a cross
  outcome = classifier.predict([instance])[0]
  markerSizeI = 12
  if outcome[0]>0.5: # class 0
    plt.plot(instance[1],instance[0],'xb', markersize=markerSizeI)
    cross = mlines.Line2D([], [], color='blue', marker='x',
                          markersize=markerSizeI-4, label='Instance')
  else:  # class 1
    plt.plot(instance[1],instance[0],'xr', markersize=markerSizeI)
    cross = mlines.Line2D([], [], color='red', marker='x',
                          markersize=markerSizeI-4, label='Instance')


  # Plot counterfactuals found
  cInstance, newClass = CF(classifier, instance, False, "distance", "CF", numTrainParam)
  cInstance2, newClass2 = protoCF(classifier, instance, False, "distance", "CF",numTrainParam)
  markerSizeC = 12
  if newClass == 0:
    plt.plot(cInstance[1],cInstance[0],'+b', markersize=markerSizeC)
    plus = mlines.Line2D([], [], color='blue', marker='+',
                          markersize=markerSizeC-4, label='Simple CF')
  else:
    plt.plot(cInstance[1],cInstance[0],'+r', markersize=markerSizeC)
    plus = mlines.Line2D([], [], color='red', marker='+',
                          markersize=markerSizeC-4, label='Simple CF')

  if newClass2 == 0:
    plt.plot(cInstance2[1],cInstance2[0],'*b', markersize=markerSizeC)
    star = mlines.Line2D([], [], color='blue', marker='*',
                          markersize=markerSizeC-4, label='Prototype CF')
  else:
    plt.plot(cInstance2[1],cInstance2[0],'*r', markersize=markerSizeC)
    star = mlines.Line2D([], [], color='red', marker='*',
                          markersize=markerSizeC-4, label='Prototype CF')

  blue = mpatches.Patch(color='blue', label='Good risk')
  red = mpatches.Patch(color='red', label='Bad risk')



  plt.legend(handles=[blue, red, cross, plus, star], fontsize=fontSize-4, loc="best")


  plt.title("Model predictions, instance to explain and counterfactuals found", fontsize=fontSize)
  plt.xlabel("Debt to Income Ratio", fontsize=fontSize)
  plt.ylabel("Number of Delinquencies", fontsize=fontSize)
  plt.show()
  plt.savefig('DB.png')


 # Generate data for correlated variables (for logistic regression and 2 features)
def correlatedData(typeCF, cFactuals, proto):
  # if proto is true, use counterfactuals guided by prototypes
    classifier = LR(2, 0.25, False, [])
    xTrain, yTrain, xTest, yTest = classifier.getUncodedData()
    # Counterfactuals are generated for xTest data
    numCEs = cFactuals
    featureVal = []
    distance = []
    timeTaken = []
    numCFVal = []
    divVal = []
    nearestTrain = []
    numTrain = []
    corr = []

    numVar = [6, 7, 8] # Numerical variables Maybe avoid 1, 2, 3
    #numVar = [1, 6, 7, 8, 11, 12] # Numerical variables
    combinations = [] # Combinations to try
    for i in numVar:
      for j in numVar:
        if j!=i and j>i:
          combinations.append([i, j])


    for item in combinations:
      featureValTemp = []
      distanceTemp = []
      timeTakenTemp = []
      numCFValTemp = []
      divValTemp = []
      nearestTrainTemp = []
      numTrainTemp = []

      classifier = LR(2, 0.25, True, item)
      correlate = classifier.correlation()
      for i in range(numCEs): # Number of simulations per couplet

          if proto:
              fVal, dist, time, numCF, diversity, nearT, numT, featChanged  = protoCF(classifier, xTest[i], False, typeCF, "data")
          else:
              fVal, dist, time, numCF, diversity, nearT, numT, featChanged = CF(classifier, xTest[i], False, typeCF, "data")

          featureValTemp.append(fVal)
          distanceTemp.append(dist)
          timeTakenTemp.append(time)
          numCFValTemp.append(numCF)
          divValTemp.append(diversity)
          nearestTrainTemp.append(nearT)
          numTrainTemp.append(numT)


      featureVal.append(np.mean(featureValTemp))
      distance.append(np.mean(distanceTemp))
      timeTaken.append(np.mean(timeTakenTemp))
      numCFVal.append(np.mean(numCFValTemp))
      divVal.append(np.mean(divValTemp))
      nearestTrain.append(np.mean(nearestTrainTemp))
      numTrain.append(np.mean(numTrainTemp))
      corr.append(correlate)

    if proto==True:
      fileName = "corrProto.csv"
    else:
      fileName = "corrSimple.csv"
    with open(fileName, "w") as csvfile:
      cols = ['features', 'distance', 'time', 'numCF', 'diversity', \
              'nearestTrain', 'numTrain', 'correlation']
      writer = csv.DictWriter(csvfile, fieldnames=cols)
      writer.writeheader()
      for i in range(len(featureVal)):
        writer.writerow({'features':featureVal[i], 'distance':distance[i], \
                         'time':timeTaken[i], 'numCF':numCFVal[i], \
                         'diversity':divVal[i], 'nearestTrain':nearestTrain[i], \
                         'numTrain':numTrain[i],
                         'correlation':corr[i]})

# Effect on correlated variables
def corrEffect():

    # Retrieves data of counterfactuals
    cols = ['features', 'distance', 'time', 'numCF', 'diversity', \
              'nearestTrain', 'numTrain', 'correlation']
    df = pd.read_csv("corrSimple.csv", usecols=cols)
    df2 = pd.read_csv("corrProto.csv", usecols=cols)
    nearT = list(df['nearestTrain'])
    nearT2 = list(df2['nearestTrain'])
    numT =list(df['numTrain'])
    numT2 = list(df2['numTrain'])
    corr =list(df['correlation'])
    featureVal = list(df["features"])
    distance = list(df['distance'])
    diversity = list(df['diversity'])
    featureVal2 = list(df2["features"])
    distance2 = list(df2['distance'])
    diversity2 = list(df2['diversity'])

    plt.figure()
    plt.plot(numT, corr, 'b')
    plt.plot(numT2, corr, 'g')
    plt.xlabel('Number of training points close to counterfactual')
    plt.ylabel('Correlation coefficient')
    plt.title('Correlation between two variables and closeness of counterfactual to data')
    plt.legend(['Simple method', 'Prototype method'])
    plt.savefig('corrNumT.png')

    plt.figure()
    plt.plot(nearT, corr, 'b')
    plt.plot(nearT2, corr, 'g')
    plt.xlabel('Distance of counterfactual to nearest data point')
    plt.ylabel('Correlation coefficient')
    plt.title('Correlation between two variables and closeness of counterfactual to nearest data')
    plt.legend(['Simple method', 'Prototype method'])
    plt.savefig('corrNearT.png')

    plt.figure()
    plt.plot(distance, corr, 'b')
    plt.plot(distance2, corr, 'g')
    plt.xlabel('Distance (Gower)')
    plt.ylabel('Correlation coefficient')
    plt.title('Distance between counterfactual and original ')
    plt.legend(['Simple method', 'Prototype method'])
    plt.savefig('corrDist.png')

    plt.figure()
    plt.plot(diversity, corr, 'b')
    plt.plot(diversity2, corr, 'g')
    plt.xlabel("Diversity (Gower distance between counterfactuals)")
    plt.ylabel('Correlation coefficient')
    plt.title('Average distance between counterfactuals generated ')
    plt.legend(['Simple method', 'Prototype method'])
    plt.savefig('corrDiv.png')

    plt.show()




# ------------------------ Main code  -----------------------------

#------------------- General simulations -------------
# 1.
l = LR(4, 0.25, False, [])
generateData(l, "distance", 100, False, 0.1) # CF
generateData(l, "distance", 100, True, 0.1) # ProtoCF

# 2. To compare 4 and 8 features
l = LR(8, 0.25, False, [])
generateData(l, "distance", 100, False, 0.1) # CF
generateData(l, "distance", 100, True, 0.1) # ProtoCF

# 3. To compare LR with DA
d = DA(4, 0.25, False, [], "LDA")
generateData(d, "distance", 100, False, 0.1) # CF
generateData(d, "distance", 100, True, 0.1) # ProtoCF

# 4.
d = DA(4, 0.25, False, [], "QDA")
generateData(d, "distance", 100, False, 0.1) # CF
generateData(d, "distance", 100, True, 0.1) # ProtoCF

#5.
d = DA(4, 0.25, False, [], "RDA")
generateData(d, "distance", 100, False, 0.1) # CF
generateData(d, "distance", 100, True, 0.1) # ProtoCF

#6. To compare features vs distance
l = LR(4, 0.25, False, [])
generateData(l, "features", 100, False, 0.1) # CF
generateData(l, "features", 100, True, 0.1) # ProtoCF

#7
l = LR(8, 0.25, False, [])
generateData(l, "features", 100, False, 0.1) # CF
generateData(l, "features", 100, True, 0.1) # ProtoCF

#--------------------- Correlation simulations -------------

#8. Compares combinations of 2 variables: uses [6, 7, 8, 11, 12]
correlatedData("distance", 20, False)

#9.
correlatedData("distance", 20, True)

#--------------------- Decision boundary -------------
l = LR(2, 0.25, False, [])
decisionBoundary(l, [3, 30], 0.1)

l = LR(2, 0.25, False, [])
decisionBoundary(l, [3, 50], 0.1)
