import pandas as pd
import numpy as np
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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=11)

class_labels = y_train.unique()
class_samples = {label: x_train[y_train == label] for label in class_labels}

conditional_probs = {}  

for label in class_labels:
    class_data = class_samples[label]
    conditional_probs[label] = {}  
    
    for feature_idx in range(len(x_train.columns)):
        feature_probs = {}  
        
        
        unique_values, value_counts = np.unique(class_data.iloc[:, feature_idx], return_counts=True)
        
       
        total_samples = len(class_data)
        for value, count in zip(unique_values, value_counts):
            probability = count / total_samples
            feature_probs[value] = probability
        
        conditional_probs[label][feature_idx] = feature_probs

class_counts = y_train.value_counts().to_dict()

predicted_labels = []
for i in range(len(x_test)):  
    test_sample = x_test.iloc[i]  
    max_posterior_prob = 0
    predicted_class = None
    
    for label in class_labels:
        prior_prob = class_counts[label] / len(y_train)
        posterior_prob = prior_prob
        
        for feature_idx, feature_value in test_sample.items():
            if feature_idx in conditional_probs[label]:
                if feature_value in conditional_probs[label][feature_idx]:
                    posterior_prob *= conditional_probs[label][feature_idx][feature_value]
        
        if posterior_prob > max_posterior_prob:
            max_posterior_prob = posterior_prob
            predicted_class = label
    
    predicted_labels.append(predicted_class)

accuracy = accuracy_score(y_test, predicted_labels)
print("Accuracy of the Bayesian classifier:", accuracy)

num_bins = 20  # You can adjust this based on your preference

# Loop through each class and its conditional probabilities
for label in class_labels:
    plt.figure(figsize=(8, 6))
    plt.title(f'Normalized Histogram of Conditional Probabilities for Class {label}')
    
    for feature_idx in range(len(x_train.columns)):
        feature_probs = list(conditional_probs[label][feature_idx].values())
        unique_values = list(conditional_probs[label][feature_idx].keys())
        
        # Normalize the probabilities to make the sum equal to 1
        total_prob = sum(feature_probs)
        normalized_probs = [prob / total_prob for prob in feature_probs]
        
        # Create the histogram
        plt.hist(unique_values, bins=num_bins, weights=normalized_probs, alpha=0.5, label=f'Feature {feature_idx}')
    
    plt.xlabel('Unique Feature Values')
    plt.ylabel('Normalized Probability')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()








# for feature_idx in range(len(x_train.columns)):

#     class_labels = [2, 4]  # Replace with the class labels you want to compare
#     class_colors = ['blue', 'red']  # Assign colors to the classes
#     hist_data = []

#     for class_label in class_labels:
#         conditional_probabilities = conditional_probs[class_label][feature_idx]
#         feature_values = list(conditional_probabilities.keys())
#         probabilities = list(conditional_probabilities.values())
        
#         # Normalize the probabilities
#         total_prob = sum(probabilities)
#         normalized_probabilities = [prob / total_prob for prob in probabilities]
        
#         hist_data.append(normalized_probabilities)

#     # Create a histogram with different colors for the two classes
#     plt.figure(figsize=(8, 6))
#     plt.hist(hist_data, bins=9, label=class_labels, color=class_colors, alpha=0.7)

#     # Set labels and title
#     plt.xlabel(f'Feature {feature_idx} Values')
#     plt.ylabel('Frequency')
#     plt.title(f'Normalized Conditional Probabilities Histogram for Feature {feature_idx}')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()














# acc_list={}
# for i in range(1,30):
#     clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=i, min_impurity_decrease=0.005)
#     clf.fit(x_train, y_train)

#     y_pred = clf.predict(x_test)

#     accuracy = accuracy_score(y_test, y_pred)
#     acc_list[i]=accuracy

#     tree_depth = clf.tree_.max_depth

# print("\n\n\nBest depth=", max(acc_list, key=acc_list.get))

# depth_values = list(acc_list.keys())
# accuracy_values = list(acc_list.values())

# # Plotting the graph
# plt.figure()
# plt.plot(depth_values, accuracy_values)
# plt.title('Depth vs Accuracy')
# plt.xlabel('Depth')
# plt.ylabel('Accuracy')
# plt.xticks(depth_values)
# plt.grid(True)
# plt.show()

# clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=max(acc_list, key=acc_list.get))
# clf.fit(x_train, y_train)
# plt.figure(figsize=(12, 8))
# plot_tree(clf, filled=True)
# plt.show()


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

