#!/usr/bin/env python
# coding: utf-8

# ## I will be doing 
# Find euclidean distances from target(unlabled iris record) to all respective labled iris records
# 
# and then will be finding( by sorting the distances)the few respective nearest neighbour and 
# 
# then will be picking the majority basis lable from those labled neighbour records to project same as lable for the 
# 
# target(unlabled iris record)



###Finding euclidean distances from unlabled target iris record to every given respective labled iris record 
def e_distance(rec1,rec2):#hear rec1 ref to labled record & rec2 to unlabled record
    distance=0
    for i in range(len(rec1)-1):
        distance=distance+(rec1[i]-rec2[i])**2
    distance=distance**0.5
    return distance




###Finding the List of nearest neighbours based on parameter of howmany to choose
def nearest_neighbours(target,labled,num_n_n):#numnn->num of nearest neighbours
    neighbours=[]
    for i in range(len(labled)):
        distance=e_distance(labled[i],target)
        #print(distance)
        neighbours.append((labled[i][-1],distance))
    ##sorting neighbours
    #neighbours
    neighbours.sort(key=lambda val:val[1])
    nearest_neighbours=[]
    nearest_neighbours.extend(neighbours[:num_n_n])
    lbls=[x for x in nearest_neighbours[0]]
    return lbls




###Now Finally choosing what would be the lable for this target
def predict_lable(nearest_neighbours):
    lbls=nearest_neighbours
    lable=max(lbls,key=lbls.count)
    return lable




###Example Iris data set
import pandas as pd
dataset1 = pd.read_csv("iris.csv")
dataset = dataset1.iloc[:,:].values



###Spliting data as test and train for classification
from sklearn.model_selection import train_test_split
x_train,x_test = train_test_split(dataset,test_size=20)



###Running the classifier 
y_test = []
y_pred = []
for i in range(len(x_test)):
    print(x_test[i][-1]+"--->"+predict_lable(nearest_neighbours(x_test[i][:-1],x_train,7)))
    y_test.append(x_test[i][-1])
    y_pred.append(predict_lable(nearest_neighbours(x_test[i][:-1],x_train,3)))



from sklearn.metrics import confusion_matrix


confusion_matrix(y_test,y_pred)


#accuracy Percentage
accuracy_score(y_test,y_pred)*100
