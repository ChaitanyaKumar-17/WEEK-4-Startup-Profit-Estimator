import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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

# Now let's do testing of the trained model along with their R2 score 
y_train_pred_lr = model.predict(X_train)
y_test_pred_lr = model.predict(X_test)

print("\n--- Multiple Linear Regression Performance ---")
print(f"Train R²: {r2_score(y_train, y_train_pred_lr):.4f} | Test R²: {r2_score(y_test, y_test_pred_lr):.4f}")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred_lr)):.2f} | Test RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred_lr)):.2f}")



# ridge_model = Ridge(alpha=10.0) # Using a higher alpha to force some regularization
# lasso_model = Lasso(alpha=100.0)

# ridge_model.fit(X_train, y_train)
# lasso_model.fit(X_train, y_train)

# y_train_pred_ridge = ridge_model.predict(X_train)
# y_test_pred_ridge = ridge_model.predict(X_test)

# y_train_pred_lasso = lasso_model.predict(X_train)
# y_test_pred_lasso = lasso_model.predict(X_test)

# print("\n--- Ridge Regression Performance ---")
# print(f"Train R²: {r2_score(y_train, y_train_pred_ridge):.4f} | Test R²: {r2_score(y_test, y_test_pred_ridge):.4f}")
# print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred_ridge)):.2f} | Test RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred_ridge)):.2f}")

# print("\n--- Lasso Regression Performance ---")
# print(f"Train R²: {r2_score(y_train, y_train_pred_lasso):.4f} | Test R²: {r2_score(y_test, y_test_pred_lasso):.4f}")
# print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred_lasso)):.2f} | Test RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred_lasso)):.2f}")

# # ==========================================
# # 3. Analyze Coefficients 
# # ==========================================
# print("\n--- Coefficient Analysis (Multiple Linear Regression) ---")
# coef_df = pd.DataFrame({
#     'Feature': X.columns, 
#     'Coefficient': model.coef_
# })
# # Sort by the absolute impact on profit
# print(coef_df.sort_values(by='Coefficient', ascending=False, key=abs))

# # ==========================================
# # 4. Plot Predicted vs Actual & Residuals
# # ==========================================
# fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# # Plot 1: Actual vs Predicted Profits (using MLR model)
# axes[0].scatter(y_test, y_test_pred_lr, color='teal', alpha=0.7)
# axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Line of perfect prediction
# axes[0].set_xlabel('Actual Profit')
# axes[0].set_ylabel('Predicted Profit')
# axes[0].set_title('Actual vs Predicted Profit (MLR)')

# # Plot 2: Residuals Distribution
# residuals = y_test - y_test_pred_lr
# sns.histplot(residuals, kde=True, ax=axes[1], color='indigo')
# axes[1].set_xlabel('Residuals (Actual - Predicted)')
# axes[1].set_title('Distribution of Residuals')

# plt.tight_layout()
# plt.show()