import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("C:/Users/syeda/OneDrive/Desktop/Datasets/water_potability.csv")
data=data.dropna()

# Separate features and labels
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into a training and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Standardize the feature data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Apply PCA to the feature data
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# Create a hard margin support vector machine classifier 
svm_hard = SVC(kernel='linear', C=float('1e6'))
svm_hard.fit(x_train_pca, y_train)
y_pred_hard = svm_hard.predict(x_test_pca)
accuracy_hard = accuracy_score(y_test, y_pred_hard)

print('Accuracy of hard margin SVM: ', accuracy_hard, sep='')

# Create a soft margin support vector machine classifier 
penalties = [0.01, 0.1, 1, 10, 100, 1000, 10000]
accuracies_soft = []

for penalty in penalties:
    svm_soft = SVC(kernel='linear', C=penalty)
    svm_soft.fit(x_train_pca, y_train)
    y_pred_soft = svm_soft.predict(x_test_pca)
    accuracy = accuracy_score(y_test, y_pred_soft)
    accuracies_soft.append(accuracy)

plt.figure()
plt.plot(penalties, accuracies_soft)
plt.title('Soft Margin SVM: Penalty vs Accuracy')
plt.xlabel('Penalty')
plt.ylabel('Accuracy')
# plt.xticks()
plt.grid(True)

best_penalty = penalties[accuracies_soft.index(max(accuracies_soft))]

# Create a polynomial kernel soft margin support vector machine classifier
polynomial_orders = [2, 3, 4]
accuracies_poly = []

for order in polynomial_orders:
    svm_poly = SVC(kernel='poly', C=best_penalty, degree=order)
    svm_poly.fit(x_train_pca, y_train)
    y_pred_poly = svm_poly.predict(x_test_pca)
    accuracy = accuracy_score(y_test, y_pred_poly)
    accuracies_poly.append(accuracy)

plt.figure()
plt.plot(polynomial_orders, accuracies_poly)
plt.title('Polynomial soft margin SVM: Polynomial Order vs Accuracy')
plt.xlabel('Order')
plt.ylabel('Accuracy')
# plt.xticks(penalties)
plt.grid(True)

# Create a radial basis function kernel soft margin support vector machine classifier
sigmas = [0.1, 1, 10]
accuracies_rbf = []

for sigma in sigmas:
    svm_rbf = SVC(kernel='rbf', C=best_penalty, gamma=sigma)
    svm_rbf.fit(x_train_pca, y_train)
    y_pred_rbf = svm_rbf.predict(x_test_pca)
    accuracy = accuracy_score(y_test, y_pred_rbf)
    accuracies_rbf.append(accuracy)

plt.figure()
plt.plot(sigmas, accuracies_rbf)
plt.title('RBF soft margin SVM: Sigma vs Accuracy')
plt.xlabel('Sigma')
plt.ylabel('Accuracy')
# plt.xticks(penalties)
plt.grid(True)
plt.show()

