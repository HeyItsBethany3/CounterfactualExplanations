#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.inspection import permutation_importance as pi
import os
import csv
from time import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
import math

# Reads in data
df = pd.read_csv(r'../data.csv')
data = np.array(df)
methodTitle = ["Logistic Regression", "Discriminant Analysis (RDA)", \
"Decision Tree", "Random Forest", "Gradient Boosting"]

# ------------------ Classes -----------------------------
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


# ------------------ Functions -----------------------------

#  Function for generating an ROC curve of all classifiers
def compareROC(features, testSize, DAType, GBMLoss, maxDepth):
    # DAType is the type of discriminant analysis, LDA, QDA or RDA
    # maxDepth is the maximum tree depth for the decision tree
    # GBMLoss is the loss function for gradient boosting: exponential or deviance

    # Construct classifiers
    l = LR(features, testSize)
    d = DA(features, testSize, DAType)
    t = Tree(features, testSize, maxDepth)
    f = Forest(features, testSize, maxDepth, 100)
    g = GBM(features, testSize, GBMLoss)

    methods = [l,d,t,f,g]
    global methodTitle
    colours = ["cornflowerblue", "darkorange", "lightcoral", "darkmagenta", "lightgreen"]
    linestyles = ["-", ":", "-", ":", "-"]
    linestyle=':'

    # Generate ROC curve
    roc = [[0]*3]*5
    j=0
    for i in methods:
        roc[j]=i.rocValues()
        j=j+1

    fig = plt.figure(figsize=(6,6))
    for i in range(len(methods)):
        plt.plot(roc[i][0], roc[i][1], color=colours[i],lw=2,linestyle=\
        linestyles[i], label='%s: %0.3f' % (methodTitle[i], roc[i][2]))

    fontSize = 12
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--', color='navy')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('ROC Curve of Classifiers in Credit Risk Prediction with 8 features', fontsize = fontSize+1)
    plt.ylabel('True Positive Rate', fontsize = fontSize)
    plt.xlabel('False Positive Rate', fontsize = fontSize)
    plt.legend(loc="lower right", title="AUC Scores")
    plt.savefig("ROC.png")


# Creates a table to compare statistics for all methods
def createTable(features, testSize, DAType, GBMLoss, maxDepth):
    os.system("rm stats.csv")
    # DAType is the type of discriminant analysis, LDA, QDA or RDA
    # maxDepth is the maximum tree depth for the decision tree
    # GBMLoss is the loss function for gradient boosting: exponential or deviance

    # Construct classifiers
    l = LR(features, testSize)
    d = DA(features, testSize, DAType)
    t = Tree(features, testSize, maxDepth)
    f = Forest(features, testSize, maxDepth, 100)
    g = GBM(features, testSize, GBMLoss)

    # Retrieves statistics for all methods
    methods = [l,d,t,f,g]
    global methodTitle
    statistics = [[0]*6]*5
    j=0
    for i in methods:
        i.generateReport()
        statistics[j]=i.statistics()
        j=j+1

    # Writes statistics to a table
    with open("stats.csv", "w") as csvfile:
        cols = ['Classification Method', 'Training Accuracy', 'Test Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=cols)

        writer.writeheader()
        k=0
        for row in statistics:
            writer.writerow({'Classification Method': methodTitle[k], \
            'Training Accuracy': row[0], 'Test Accuracy': \
            row[1]})

            """
            writer.writerow({'Classification method': methodTitle[k], \
            'Training accuracy': row[0], 'Test accuracy': \
            row[1], 'Precision 0': row[2], 'Recall 0': row[3], \
            'Precision 1': row[4], 'Recall 1': row[5]})
            """
            k=k+1


# Creates plots of actual and predicted instances for all methods
def createPlots(testSize, DAType, GBMLoss, maxDepth):
    # DAType is the type of discriminant analysis, LDA, QDA or RDA
    # maxDepth is the maximum tree depth for the decision tree
    # GBMLoss is the loss function for gradient boosting: exponential or deviance

    # Construct classifiers
    l = LR(2, testSize)
    d = DA(2, testSize, DAType)
    t = Tree(2, testSize, maxDepth)
    f = Forest(2, testSize, maxDepth, 100)
    g = GBM(2, testSize, GBMLoss)

    # Creates large plot
    fig, ((ax1, ax2, ax3, ax4, ax5, ax6), (ax7, ax8, ax9, ax10, ax11, ax12))\
     = plt.subplots(nrows=2, ncols=6, figsize=(14, 5))
    plots = ((ax2, ax3, ax4, ax5, ax6), (ax8, ax9, ax10, ax11, ax12))

    methods = [l,d,t,f,g]
    global methodTitle
    instances = [[0]*2]*5
    j=0
    for i in methods:
        instances[j]=i.getPredictions()
        j=j+1
    xTrain, yTrain, xTest, yTest = l.getData()

    for i in range(len(yTrain)):
        if yTrain[i]==0:
            # Good credit risk - blue
            ax1.plot(xTrain[i][1],xTrain[i][0],'ob', markersize=2)
        else:
            # Bad credit risk - red
            ax1.plot(xTrain[i][1],xTrain[i][0],'or', markersize=2)

    for i in range(len(yTest)):
        if yTest[i]==0:
            ax7.plot(xTest[i][1], xTest[i][0],'ob', markersize=2)
        else:
            ax7.plot(xTest[i][1], xTest[i][0],'or', markersize=2)

    ax1.set_title("Training Data")
    ax7.set_title("Test Data")
    ax1.set_xlabel("Debt-Income Ratio")
    ax1.set_ylabel("Number of Delinquencies")
    ax7.set_xlabel("Debt-Income Ratio")
    ax7.set_ylabel("Number of Delinquencies")


    j=0
    for image in plots[0]:
        for i in range(len(xTrain)):
            if instances[j][0][i]==0:
                image.plot(xTrain[i][1], xTrain[i][0],'ob', markersize=2)
            else:
                image.plot(xTrain[i][1], xTrain[i][0],'or', markersize=2)
        image.set_title("{}".format(methodTitle[j]))
        j=j+1

    k=0
    for image in plots[1]:
        for i in range(len(xTest)):
            if instances[k][1][i]==0:
                image.plot(xTest[i][1], xTest[i][0],'ob', markersize=2)
            else:
                image.plot(xTest[i][1], xTest[i][0],'or', markersize=2)
        k=k+1

    plt.savefig("PredictedData.png")


# Shows the accuracy of all classifiers
def showAllAccuracy(features, testSize, DAType, GBMLoss, maxDepth):
    # DAType is the type of discriminant analysis, LDA, QDA or RDA
    # maxDepth is the maximum tree depth for the decision tree
    # GBMLoss is the loss function for gradient boosting: exponential or deviance

    l = LR(features, testSize)
    d = DA(features, testSize, DAType)
    t = Tree(features, testSize, maxDepth)
    f = Forest(features, testSize, maxDepth, 100)
    g = GBM(features, testSize, GBMLoss)

    methods = [l,d,t,f,g]
    global methodTitle
    j=0
    for i in methods:
        print("methodTitle[j]: ", end="")
        i.showAccuracy()
        j=j+1

# ------------------- Main code -----------------

compareROC(8, 0.25, 'RDA','exponential', 5)
