import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_excel("C:/Users/syeda/OneDrive/Desktop/Datasets/DryBeanDataset/Dry_Bean_Dataset.xlsx")
sns.pairplot(df, hue='Class')

plt.show()