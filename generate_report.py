import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    df = pd.read_csv(os.path.join(BASE_DIR, 'preprocessed_data.csv'))
    print("Data loaded successfully.")
except FileNotFoundError:
    print("preprocessed_data.csv not found. Please run preprocess_data.py to generate it first.")
    exit(1)

print("Generating Feature Distributions...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['total_311'], kde=True, ax=axes[0], color='blue')
axes[0].set_title('Distribution of Monthly 311 Complaints')
sns.histplot(df['unresolved_proportion'], kde=True, ax=axes[1], color='orange')
axes[1].set_title('Distribution of Unresolved Proportion')
sns.histplot(df['total_crimes'], kde=True, ax=axes[2], color='red')
axes[2].set_title('Distribution of Monthly Crimes')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_distributions.png'))
plt.close()

print("Generating Correlation Matrix...")
plt.figure(figsize=(8, 6))
corr_matrix = df[['total_311', 'unresolved_311', 'unresolved_proportion', 'total_crimes']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'))
plt.close()

print("Generating Bivariate Analysis...")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_311', y='total_crimes', size='unresolved_proportion', hue='unresolved_proportion', sizes=(20, 200), palette='viridis', data=df)
plt.title('Monthly 311 Complaint Volume vs. Monthly Crime Volume by Precinct')
plt.savefig(os.path.join(OUTPUT_DIR, 'bivariate_analysis.png'))
plt.close()

print("Running Hypothesis Testing...")
correlation, p_value = stats.pearsonr(df['total_311'], df['total_crimes'])
with open(os.path.join(OUTPUT_DIR, 'hypothesis_test_results.txt'), 'w') as f:
    f.write(f"Pearson Correlation Coefficient: {correlation:.4f}\n")
    f.write(f"P-value: {p_value:.4e}\n")
    if p_value < 0.05:
        f.write("Result: Reject Null Hypothesis. Statistically significant relationship found.\n")
    else:
        f.write("Result: Fail to Reject Null Hypothesis.\n")

print("Running Model Building (Random Forest Regressor)...")
X = df[['total_311', 'unresolved_proportion']]
y = df['total_crimes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)

kf = KFold(n_splits=5, shuffle=True, random_state=0)
cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

with open(os.path.join(OUTPUT_DIR, 'model_metrics.txt'), 'w') as f:
    f.write(f"--- 5-Fold Cross Validation (Training Data) ---\n")
    f.write(f"CV R^2 Scores: {cv_scores}\n")
    f.write(f"Mean CV R^2: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n\n")
    f.write(f"--- Holdout Test Set Performance ---\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R^2 Score: {r2:.4f}\n")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Monthly Crime Volume')
plt.ylabel('Predicted Monthly Crime Volume')
plt.title('Random Forest Regression: Actual vs Predicted Monthly Crime Volume')
plt.savefig(os.path.join(OUTPUT_DIR, 'model_predictions.png'))
plt.close()

print("All reports and visualizations generated successfully in the 'output' folder.")
