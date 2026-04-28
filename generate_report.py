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

# Set a consistent visual theme across all charts
sns.set_theme(style='darkgrid', palette='viridis')

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
plt.title('Feature Correlation Matrix (total_311 is the strongest predictor of crime)')
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'))
plt.close()

print("Generating Bivariate Analysis...")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_311', y='total_crimes', size='unresolved_proportion', hue='unresolved_proportion', sizes=(20, 200), palette='viridis', data=df)
plt.title('Monthly 311 Complaint Volume vs. Monthly Crime Volume by Precinct')
plt.xlabel('Monthly 311 Complaints')
plt.ylabel('Monthly Crime Volume')
plt.savefig(os.path.join(OUTPUT_DIR, 'bivariate_analysis.png'))
plt.close()

print("Generating Time-Series Chart...")
time_df = df.groupby('YearMonth')[['total_311', 'total_crimes']].mean().reset_index()
time_df = time_df.sort_values('YearMonth')
fig, ax1 = plt.subplots(figsize=(14, 6))
color1 = 'tab:blue'
ax1.set_xlabel('Year-Month')
ax1.set_ylabel('Avg Monthly 311 Complaints', color=color1)
ax1.plot(time_df['YearMonth'], time_df['total_311'], color=color1, lw=2)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.tick_params(axis='x', rotation=90)
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Avg Monthly Crime Volume', color=color2)
ax2.plot(time_df['YearMonth'], time_df['total_crimes'], color=color2, lw=2, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color2)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, ['Avg 311 Complaints', 'Avg Crime Volume'], loc='upper left')

plt.title('Monthly Trends: 311 Complaints vs Crime Volume (Avg Across All Precincts)')
fig.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'time_series.png'))
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

rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # np.sqrt used for sklearn 1.8+ compatibility
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

print("Generating Residual Plot...")
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5, color='purple')
plt.axhline(y=0, color='red', linestyle='--', lw=2)
plt.xlabel('Predicted Monthly Crime Volume')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot: Checking for Model Bias')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'residual_plot.png'))
plt.close()

print("Generating Feature Importance Chart...")
importances = model.feature_importances_
feature_names = ['Total 311 Volume', 'Unresolved Proportion']
sorted_idx = np.argsort(importances)[::-1]
plt.figure(figsize=(8, 5))
sns.barplot(x=[importances[i] for i in sorted_idx], y=[feature_names[i] for i in sorted_idx], palette='coolwarm')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.title('Feature Importance: What Drives Crime Predictions?')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
plt.close()

print("\n" + "="*55)
print("  URBAN PULSE NYC — PIPELINE RESULTS SUMMARY")
print("="*55)
print(f"  Pearson Correlation (311 vs Crime): r = {correlation:.4f}")
if p_value < 0.05:
    print(f"  Hypothesis Test:  H0 REJECTED  (p = {p_value:.2e})")
else:
    print(f"  Hypothesis Test:  Failed to reject H0  (p = {p_value:.2e})")
print(f"  Mean CV R\u00b2 Score:  {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"  Holdout Test R\u00b2: {r2:.4f}")
print(f"  RMSE:             {rmse:.2f} crimes/month")
print(f"  Top Predictor:    {feature_names[sorted_idx[0]]} ({importances[sorted_idx[0]]:.1%} importance)")
print("="*55)
print("All reports and visualizations saved to the 'output' folder.")

