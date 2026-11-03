X = df.copy()
y = df['remaining_time (y)']

num_cols = ['case:Amount', 'time_prev', 'time_start', 'step_counter', 'day of week']
cat_cols = ['org:role', 'org:resource', 'concept:name']

X_onehot = (df[['org:role', 'org:resource', 'concept:name']])
X_onehot = (pd.get_dummies(X_onehot, columns=['org:role', 'org:resource', 'concept:name'], drop_first=True))
X_unified = pd.concat([df[num_cols], X_onehot], axis=1, join='inner')

# %%
# 1. Feature Engineering: Include trace prefix activity counts
activity_dummies = pd.get_dummies(df['concept:name'], prefix='act')
prefix_counts = activity_dummies.groupby(df['case:DeclarationNumber']).cumsum()

# Combine numeric features, one-hot current event features, and prefix history
X_unified = pd.concat([df[num_cols], X_onehot, prefix_counts], axis=1)
y = df['remaining_time (y)']

# 2. Group Split by Case
groups = df['case:DeclarationNumber']
gss = GroupShuffleSplit(n_splits=4, train_size=0.7, random_state=42)
train_idx, test_idx = next(gss.split(X_unified, y, groups=groups))


import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

results = []

models = {
    "Ridge Regression": (Ridge(alpha=100.0, random_state=42), True),
    "Random Forest":    (RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1), False),
    "HistGradientBoosting": (HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=42), False),
}

for name, (model, use_scaled) in models.items():
    print(f'Training {name}')

    for fold, (train_idx, test_idx) in enumerate(gss.split(X_unified, y, groups=groups)):
        
        # 1. Slice data (adjust names to match your scaled/unscaled variables)
        X_tr = X_unified.iloc[train_idx]
        X_te = X_unified.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 2. Fit and predict
        model.fit(X_tr, y_train)
        y_pred = np.clip(model.predict(X_te), a_min=0, a_max=None)

        # 3. Evaluate
        results.append({
            "Model": name,
            "Fold": fold,
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": root_mean_squared_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred)
        })

print(pd.DataFrame(results).to_string(index=False))





