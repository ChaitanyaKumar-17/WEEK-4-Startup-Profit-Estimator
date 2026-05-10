import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
url = "https://raw.githubusercontent.com/krishnaik06/Multiple-Linear-Regression/master/50_Startups.csv"
df = pd.read_csv(url)
print(df.head())

# Checking the dataset
print(df.info())

# Let's handle the 'State' feature since it is a numerical column
encoder = OneHotEncoder()
encoded = encoder.fit_transform(df[['State']])
encoded_df = pd.DataFrame(encoded.toarray(), columns = encoder.get_feature_names_out())
df = pd.concat([df, encoded_df], axis = 1)
df.drop(['State'], axis = 1, inplace = True)
print(df.head())

# Now let's split the test and training data
y = df['Profit']
X = df.drop(['Profit'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Let's visualize the training dataset
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: R&D vs Profit
sns.scatterplot(data=df, x='R&D Spend', y='Profit', ax=axes[0])
axes[0].set_title('R&D Spend vs Profit')

# Plot 2: Administration vs Profit
sns.scatterplot(data=df, x='Administration', y='Profit', ax=axes[1])
axes[1].set_title('Administration vs Profit')

# Plot 3: Marketing Spend vs Profit
sns.scatterplot(data=df, x='Marketing Spend', y='Profit', ax=axes[2])
axes[2].set_title('Marketing Spend vs Profit')

plt.tight_layout()
plt.show()

# Now let's do the cross-validation of the training dataset 
model = LinearRegression()

cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average CV Score: {np.mean(cv_scores):.4f}")

# Now let's fit the model
model.fit(X_train, y_train)