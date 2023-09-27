import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

filepath="C:/Users/syeda/OneDrive/Desktop/Datasets/breast-cancer-wisconsin.data"

df= pd.read_csv(filepath, delimiter=',', header=None)
df=df.drop(columns=0,axis=1)
df.replace('?', np.nan, inplace=True)
new_df=df.dropna()

x=new_df.drop(colums=6)
y=new_df[10]

list_of_error_dict_dt=[]
list_of_error_dict_knn=[]
k_val=[3,5,7,9,11,13,15]

for rsv in range(1,11):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=rsv)
    error_list_dt={}
    error_list_knn={}

    for k in k_val:
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(x_train, y_train)
        y_pred = knn_classifier.predict(x_test)
        accuracy_knn = accuracy_score(y_test, y_pred)
        error_list_knn[k]=1-accuracy_knn

    list_of_error_dict_knn.append(error_list_knn)

    for i in range(1,8):
        clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=i)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy_dt = accuracy_score(y_test, y_pred)
        error_list_dt[i]=1-accuracy_dt

    list_of_error_dict_dt.append(error_list_dt)


average_err_dt={}
sum=0
for depth in range(1,8):
    for dict in list_of_error_dict_dt:
        if depth in dict:
            sum+=dict[depth]
    average_err_dt[depth]=(sum/len(list_of_error_dict_dt))
    sum=0
depth_values = list(average_err_dt.keys())
accuracy_values = list(average_err_dt.values())
# Plotting the graph
plt.figure()
plt.plot(depth_values, accuracy_values)
plt.title('Depth vs Error')
plt.xlabel('Max Depth')
plt.ylabel('Error')
plt.xticks(depth_values)
plt.grid(True)
plt.show()



average_err_knn={}
sum=0
for k in k_val:
    for dict in list_of_error_dict_knn:
        if k in dict:
            sum+=dict[k]
    average_err_knn[k]=(sum/len(list_of_error_dict_knn))
    sum=0
k_values = list(average_err_knn.keys())
accuracy_values = list(average_err_knn.values())
# Plotting the graph
plt.figure()
plt.plot(k_values, accuracy_values)
plt.title('K vs Error')
plt.xlabel('K')
plt.ylabel('Error')
plt.xticks(k_values)
plt.grid(True)
plt.show()
