import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

# -----------------------------
# 1. Load & Preprocess Users and Merchants
# -----------------------------

users = pd.read_csv('users.csv')
users['user_country'] = users['country']
users.drop(columns=['country'], inplace=True)
users['user_index'] = users.index

merchants = pd.read_csv('merchants.csv')
merchants['merchant_country'] = merchants['country']
merchants.drop(columns=['country'], inplace=True)
merchants['merchant_index'] = merchants.index

# -----------------------------
# 2. Load Transactions
# -----------------------------

with open('transactions.json', 'r') as f:
    transactions = pd.DataFrame([json.loads(line) for line in f])

# -----------------------------
# 3. Merge All Data
# -----------------------------

full_data = (
    transactions
    .merge(users, on='user_id', how='left')
    .merge(merchants, on='merchant_id', how='left')
)

# -----------------------------
# 4. Basic Cleaning & Feature Engineering
# -----------------------------

full_data['education'] = full_data['education'].fillna('Missing')

full_data['fraud_transactions_per_merchant'] = full_data.groupby('merchant_id')['is_fraud'].transform('sum')
full_data['transactions_per_merchant'] = full_data.groupby('merchant_id')['is_fraud'].transform('count')
full_data['fraud_transactions_per_user'] = full_data.groupby('user_id')['is_fraud'].transform('sum')
full_data['transactions_per_user'] = full_data.groupby('user_id')['is_fraud'].transform('count')

full_data['fraud_ratio_per_merchant'] = full_data['fraud_transactions_per_merchant'] / full_data['transactions_per_merchant']
full_data['fraud_ratio_per_user'] = full_data['fraud_transactions_per_user'] / full_data['transactions_per_user']

# Drop intermediates
full_data.drop(columns=['fraud_transactions_per_user', 'fraud_transactions_per_merchant'], inplace=True)

# Time-based features
full_data['timestamp'] = pd.to_datetime(full_data['timestamp'])
full_data['signup_date'] = pd.to_datetime(full_data['signup_date'])

full_data['hour'] = full_data['timestamp'].dt.hour
full_data['day_of_week'] = full_data['timestamp'].dt.dayofweek
full_data['days_since_signup'] = (full_data['timestamp'] - full_data['signup_date']).dt.days

# Amount ratios
user_avg = full_data.groupby('user_id')['amount'].transform('mean')
full_data['amount_to_user_avg'] = full_data['amount'] / (user_avg + 1e-5)

# Sort before rolling features
full_data = full_data.sort_values(['user_id', 'timestamp'])

# -----------------------------
# 5. Rolling Transaction Count (last 7 days)
# -----------------------------

def rolling_txn_count(df):
    df = df.set_index('timestamp')
    return df['transaction_id'].rolling('7d').count()

full_data['txn_count_last_7d'] = (
    full_data
    .groupby('user_id', group_keys=False)
    .apply(rolling_txn_count)
    .reset_index(drop=True)
)

# -----------------------------
# 6. Categorical Encoding
# -----------------------------

# One-hot encode major categorical features
sex_dummies       = pd.get_dummies(full_data['sex'], prefix='sex')
education_dummies = pd.get_dummies(full_data['education'], prefix='education')
income_dummies    = pd.get_dummies(full_data['primary_source_of_income'], prefix='primary_income')
category_dummies  = pd.get_dummies(full_data['category'], prefix='category')
user_country_dummies = pd.get_dummies(full_data['user_country'], prefix='user_country')
merchant_country_dummies = pd.get_dummies(full_data['merchant_country'], prefix='merchant_country')

# Merge all dummies
full_data = pd.concat([
    full_data.drop(['category', 'user_country', 'merchant_country'], axis=1),
    sex_dummies,
    education_dummies,
    income_dummies,
    category_dummies,
    user_country_dummies,
    merchant_country_dummies
], axis=1)

# -----------------------------
# 7. Additional Features
# -----------------------------

# First transaction flag
full_data['is_first_ever_transaction'] = (
    full_data.groupby('user_id').cumcount() == 0
).astype(int)

# Ratios
full_data['expense_income_ratio'] = full_data['sum_of_monthly_expenses'] / (full_data['sum_of_monthly_installments'] + 1e-5)
full_data['txn_to_expense_ratio'] = full_data['amount'] / (full_data['sum_of_monthly_expenses'] + 1e-5)

# Frequency encodings
for col in ['channel', 'device', 'payment_method', 'education']:
    freq = full_data[col].value_counts(normalize=True)
    full_data[f'{col}_freq'] = full_data[col].map(freq)

# -----------------------------
# 8. Correlation Filtering and Heatmap
# -----------------------------

# Select numeric and boolean columns
numeric_data = full_data.select_dtypes(include=['number', 'bool'])

# Compute correlation matrix
corr_matrix = numeric_data.corr()

# Set thresholds
threshold_general = 1       # general correlation threshold
threshold_target = 0.03     # correlation with is_fraud

# Remove diagonal self-correlation
corr_no_diag = corr_matrix.copy()
np.fill_diagonal(corr_no_diag.values, 0)

# Apply filtering conditions
condition_general = (np.abs(corr_no_diag) >= threshold_general).any(axis=0)
condition_target = np.abs(corr_matrix['is_fraud']) >= threshold_target
keep_columns = condition_general | condition_target

# Filter the correlation matrix
filtered_corr = corr_matrix.loc[keep_columns, keep_columns]

# Plot
plt.figure(figsize=(14, 10))
sns.heatmap(
    filtered_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.8}
)
plt.title(f'Filtered Correlation Matrix (Thresholds: General ≥ {threshold_general}, is_fraud ≥ {threshold_target})',
          pad=20, fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
# plt.show()

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score,
    recall_score, roc_curve, auc, accuracy_score
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------------
# 9. Feature Selection
# -----------------------------

target = 'is_fraud'
correlations = full_data.select_dtypes(include=[np.number]).corr()[target].drop(target)
selected_features = correlations[abs(correlations) >= 0].index.tolist()

X = full_data[selected_features]
y = full_data[target]

# -----------------------------
# 10. Train/Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# -----------------------------
# 11. Train XGBoost Model
# -----------------------------

params = {
    'objective':        'binary:logistic',
    'eval_metric':      'auc',
    'scale_pos_weight': 12.5,
    'learning_rate':    0.01,
    'n_estimators':     500,
    'max_depth':        6,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'random_state':     42,
}

model = XGBClassifier(**params)
model.fit(X_train, y_train)

# -----------------------------
# 12. Predict Probabilities
# -----------------------------

y_pred_prob = model.predict_proba(X_test)[:, 1]

# -----------------------------
# 13. Threshold Optimization
# -----------------------------

def find_best_threshold(y_true, y_pred_prob, metric='f1', step=0.01, verbose=False):
    best_score = -1
    best_threshold = 0.5
    thresholds = np.arange(0.0, 1.01, step)

    for threshold in thresholds:
        y_pred = (y_pred_prob >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        elif metric == 'balanced':
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            score = 2 * (precision * recall) / (precision + recall + 1e-9)
        else:
            raise ValueError("Invalid metric. Choose from 'f1', 'precision', 'recall', 'balanced'.")

        if verbose:
            print(f"Threshold: {threshold:.2f} | {metric}: {score:.4f}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score

best_thresh, best_f1 = find_best_threshold(y_test, y_pred_prob, metric='f1', verbose=True)
print(f"\nBest Threshold: {best_thresh:.2f} with F1 score: {best_f1:.4f}")

# Apply best threshold
y_pred = (y_pred_prob >= best_thresh).astype(int)

# -----------------------------
# 14. Evaluation
# -----------------------------

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_prob):.4f}")

# -----------------------------
# 15. ROC Curve
# -----------------------------

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# -----------------------------
# 16. Confusion Matrix Plot
# -----------------------------

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(y_test, y_pred)

# -----------------------------
# 17. Feature Importances
# -----------------------------

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), np.array(selected_features)[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping

# # Normalize input data
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Define NN architecture
# nn_model = Sequential([
#     Dense(128, input_dim=X_train.shape[1], activation='relu'),
#     BatchNormalization(),
#     Dropout(0.3),
#     Dense(64, activation='relu'),
#     Dropout(0.3),
#     Dense(1, activation='sigmoid')
# ])

# # Compile the model
# nn_model.compile(
#     loss='binary_crossentropy',
#     optimizer=Adam(learning_rate=0.001),
#     metrics=['accuracy']
# )

# # Train the model
# early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# history = nn_model.fit(
#     X_train_scaled, y_train,
#     validation_split=0.2,
#     epochs=50,
#     batch_size=64,
#     callbacks=[early_stop],
#     verbose=1
# )

# # Evaluate
# y_pred_prob_nn = nn_model.predict(X_test_scaled).ravel()
# y_pred_nn = (y_pred_prob_nn >= 0.5).astype(int)

# print("\nNeural Network Results:")
# print(f"Accuracy: {accuracy_score(y_test, y_pred_nn):.4f}")
# print(f"ROC AUC:  {roc_auc_score(y_test, y_pred_prob_nn):.4f}")

# # Confusion Matrix
# plot_confusion_matrix(y_test, y_pred_nn)
# # ROC Curve
# fpr_nn, tpr_nn, _ = roc_curve(y_test, y_pred_prob_nn)
# roc_auc_nn = auc(fpr_nn, tpr_nn)
# plt.figure(figsize=(10, 6))
# plt.plot(fpr_nn, tpr_nn, color='blue', label=f'ROC curve (area = {roc_auc_nn:.2f})')    
# plt.plot([0, 1], [0, 1], color='red', linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (NN)')
# plt.legend(loc='lower right')
# plt.tight_layout()
# plt.show()