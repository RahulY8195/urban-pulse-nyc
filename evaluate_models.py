import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    df = pd.read_csv(os.path.join(BASE_DIR, 'preprocessed_data.csv'))
except FileNotFoundError:
    print("Error: preprocessed_data.csv not found. Please run preprocess_data.py first.")
    exit(1)

X = df[['total_311', 'unresolved_proportion']]
y = df['total_crimes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
kf = KFold(n_splits=5, shuffle=True, random_state=0)

regression_models = {
    'Linear Regression': (LinearRegression(), {}),
    'KNN': (KNeighborsRegressor(), {'n_neighbors': [3, 5, 10, 20], 'weights': ['uniform', 'distance']}),
    'Decision Tree': (DecisionTreeRegressor(random_state=0), {'max_depth': [None, 3, 5], 'min_samples_split': [2, 5]}),
    'Random Forest': (RandomForestRegressor(random_state=0), {'n_estimators': [50, 100], 'max_depth': [None, 3, 5]}),
    'SVM (SVR)': (SVR(), {}) 
}

model_labels = []
scores_list = []

print("Running Hyperparameter Model Selection (this may take 20-30 seconds)...")

for name, (model, params) in regression_models.items():
    grid = GridSearchCV(model, params, cv=kf, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    score = grid.best_score_
    best_params = grid.best_params_
    
    if not best_params:
        label = f"{name}\n(Baseline)"
    else:
        param_str = ", ".join([f"{k}={v}" for k, v in best_params.items()])
        label = f"{name}\n({param_str})"
        
    model_labels.append(label)
    scores_list.append(score)
    print(f"Tuned {name}: R2 = {score:.4f} | Params: {best_params if best_params else '(Baseline — no tuning applied)'}")

plt.figure(figsize=(12, 8))

sorted_indices = np.argsort(scores_list)[::-1]
sorted_labels = [model_labels[i] for i in sorted_indices]
sorted_scores = [scores_list[i] for i in sorted_indices]

plot_scores = [max(0, s) for s in sorted_scores] 

sns.barplot(x=plot_scores, y=sorted_labels, palette='viridis')
plt.title('Hyperparameter Model Selection: Cross-Validation R² Scores', fontsize=16, fontweight='bold')
plt.xlabel('Mean CV R² Score (Higher is Better)', fontsize=12)
plt.ylabel('Algorithm (Optimal Hyperparameters)', fontsize=12)

for index, value in enumerate(sorted_scores):
    if value < 0:
        plt.text(0.01, index, f' FAILED (R² = {value:.4f})', va='center', color='red', fontweight='bold', fontsize=11)
    else:
        plt.text(value, index, f' {value:.4f}', va='center', fontweight='bold', fontsize=11)

if max(plot_scores) > 0:
    plt.xlim(0, max(plot_scores) + 0.05)

plt.tight_layout()
output_file = os.path.join(OUTPUT_DIR, 'model_comparison_bar.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Successfully generated hyperparameter selection chart at: {output_file}")
