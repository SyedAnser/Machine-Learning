import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df=pd.read_excel("C:/Users/syeda/OneDrive/Desktop/Datasets/Raisin_Dataset/Raisin_Dataset.xlsx")

df.replace('?', np.nan, inplace=True)
new_df=df.dropna()
x=new_df.drop(columns='Class')
y=new_df['Class']

split_val=np.array_split(x,10)
split_class=np.array_split(y,10)
list_acc=[]

for i in range(10):
    x_test=split_val.pop(i)
    x_train=np.concatenate(split_val)
    y_test=split_class.pop(i)
    y_train=np.concatenate(split_class)
    acc={}
    for j in range(2,9):
        clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=j)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc[j] = accuracy_score(y_test, y_pred)
    list_acc.append(acc)
    split_val=np.array_split(x,10)
    split_class=np.array_split(y,10)


sum=0
avg_acc={}
sd=[]
for depth in range(2,9):
    for dict in list_acc:
        if depth in dict:
            sum+=dict[depth]
    avg_acc[depth]=(sum/len(list_acc))
    sum=0
depth_values = list(avg_acc.keys())
accuracy_values = list(avg_acc.values())

temp=[]
for depth in range(2,9):
    for dict in list_acc:
        if depth in dict:
            temp.append(dict[depth])
    sd.append(np.std(temp))

print(sd)
# Plotting the graph
plt.figure()
plt.errorbar(depth_values, accuracy_values, yerr=sd)
plt.title('Depth vs Accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.xticks(depth_values)
plt.grid(True)

plt.show()