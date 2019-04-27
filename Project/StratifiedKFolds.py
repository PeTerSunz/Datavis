__author__ = 'ChatchaiKASEMTAWEECH'


__author__ = 'Lenovo'
from math import exp, expm1
from random import randint,random
from pandas import DataFrame, read_csv
from numpy import linalg as LA
from numpy import random
from sklearn.model_selection import StratifiedKFold
from operator import itemgetter
import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
import time
import math
import os
import re

INPUT_FILE_LOCATION1 = 'E:\\CLASS\\01418324\\tips.csv'
INPUT_FILE_LOCATION2 = 'E:\\CLASS\\01418324\\tips.csv'
OUTPUT_FILE_LOCATION1 = 'E:\\CLASS\\01418324\\tips_xxtra.csv'
OUTPUT_FILE_LOCATION2 = 'E:\\CLASS\\01418324\\tips_xxtst.csv'
NUMBER_OF_LABEL =  1 #29
NUMBER_OF_FOLD = 5

data = read_csv(INPUT_FILE_LOCATION1)                            # read data from CSV file
df = pd.DataFrame(data = data)                                     # create dataframe from CSV data                                             # create dataframe from CSV data (original data)                      # Convert Categorical data to Numeric data in each column in data table
columnlist = list(df.columns.values)
skf = StratifiedKFold(n_splits=NUMBER_OF_FOLD)
colnum = len(columnlist)

# Split attributes (X) and class (y)
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
FoldNo = 0

# Iterates training data and test data in k-folds.
for train_index, test_index in skf.split(X, y):
    start_time = time.time()
    FoldNo += 1
    print("TRAIN:", train_index, "TEST:", test_index)

    X_train = df.ix[train_index].values
    X_test = df.ix[test_index].values

    # create training data file and test data file name
    OUTPUT_FILE_LOCATION11 = OUTPUT_FILE_LOCATION1.replace("xx",str(FoldNo))
    OUTPUT_FILE_LOCATION21 = OUTPUT_FILE_LOCATION2.replace("xx",str(FoldNo))

    ''' Create Train Data File '''
    f = open(OUTPUT_FILE_LOCATION11, 'w')
    colno = 0
    for item in columnlist:
        colno += 1
        if colno < len(columnlist):
            f.write("%s," % item)
        else:
            f.write("%s" % item)
    f.write("\n");
    f.close()

    with open(OUTPUT_FILE_LOCATION11,'ab') as f:                        # Open output file in append binary mode
        writer = csv.writer(f, delimiter=',',quoting=csv.QUOTE_MINIMAL)
        writer.writerows(X_train)                                      # write all represetative points
        f.close()



    ''' Create Test Data File '''
    f = open(OUTPUT_FILE_LOCATION21, 'w')
    colno = 0
    for item in columnlist:
        colno += 1
        if colno < len(columnlist):
            f.write("%s," % item)
        else:
            f.write("%s" % item)
    f.write("\n");
    f.close()

    with open(OUTPUT_FILE_LOCATION21,'ab') as f:                        # Open output file in append binary mode
        writer = csv.writer(f, delimiter=',',quoting=csv.QUOTE_MINIMAL)
        writer.writerows(X_test)                                      # write all represetative points
        f.close()

    print  "%s is created successfully." % OUTPUT_FILE_LOCATION11
    print  "%s is created successfully." % OUTPUT_FILE_LOCATION21
print "All output files are already created..."

process_time = (time.time() - start_time)
hour = int(process_time/3600)
minute = int((process_time - (hour *3600))/60)
second = int(process_time - (hour *3600) - (minute * 60))
print (NUMBER_OF_FOLD + " fold-cross validation files processing time --- " + str(hour) + ":" + str(minute) + ":" + str(second))




