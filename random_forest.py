#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


digits = load_digits()
X, y = digits.data, digits.target


# In[3]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.90)


# In[4]:


clf = RandomForestClassifier(n_estimators=200)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = clf.score(X_test,y_test)
print(accuracy)


# In[5]:


accuracies_foret = list()
for i in range(0,99) : 
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.90)
    clf = RandomForestClassifier(n_estimators=200)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test,y_test)
    accuracies_foret.append(accuracy)


# In[6]:


plt.plot(accuracies_foret)


# In[7]:


std_accuracy = np.std(accuracies_foret)
std_accuracy


# In[8]:


accuracies_liste = list()
liste = list()
for i in range(10,1000,20):
    clf = RandomForestClassifier(n_estimators=i)
    clf.fit(X_train, y_train)
    Z = clf.predict(X_test)
    accuracy = clf.score(X_test,y_test)
    accuracies_liste.append(accuracy)
    liste.append(i)


# In[9]:


plt.plot(liste,accuracies_liste)


# In[10]:


clf = ExtraTreesClassifier(n_estimators=200)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = clf.score(X_test,y_test)
print(accuracy)


# In[11]:


accuracies_foret_extra = list()
for i in range(0,99) : 
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.90)
    clf = ExtraTreesClassifier(n_estimators=200)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test,y_test)
    accuracies_foret_extra.append(accuracy)


# In[ ]:




