import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

filepath="C:/Users/syeda/OneDrive/Desktop/Datasets/breast-cancer-wisconsin.data"

df= pd.read_csv(filepath, delimiter=',', header=None)
df=df.drop(columns=0,axis=1)
df.replace('?', np.nan, inplace=True)
new_df=df.dropna()

x=new_df.drop(columns=10)
y=new_df[10]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

acc_list={}
for i in range(1,30):
    clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=i, min_impurity_decrease=0.005)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    acc_list[i]=accuracy

    tree_depth = clf.tree_.max_depth

print("\n\n\nBest depth=", max(acc_list, key=acc_list.get))

depth_values = list(acc_list.keys())
accuracy_values = list(acc_list.values())

# Plotting the graph
plt.figure()
plt.plot(depth_values, accuracy_values)
plt.title('Depth vs Accuracy')
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.xticks(depth_values)
plt.grid(True)
plt.show()

clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=max(acc_list, key=acc_list.get))
clf.fit(x_train, y_train)
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True)
plt.show()








# for feature_column in x_train.columns:
#     clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=3)  # Example parameters
#     clf.fit(x_train[[feature_column]], y_train)
#     univariate_trees.append(clf)

# accuracies = []
# for idx, tree in enumerate(univariate_trees):
#     y_pred = tree.predict(x_test[[x_test.columns[idx]]])
#     accuracy = accuracy_score(y_test, y_pred)
#     accuracies.append(accuracy)
#     print(f"Accuracy for feature '{x_test.columns[idx]}': {accuracy:.2f}")

# train_df, test_df= train_test_split(df, test_size=0.3, random_state=1)
# print("Training Set:", train_df)
# print("Test Set:", test_df)

# sns.pairplot(train_df, hue=10)
# import matplotlib.pyplot as plt
# plt.show()

