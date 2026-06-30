### Data Loading & Inspection
import pandas as pd
import numpy as np

# Load
df = pd.read_csv("data.csv")            # also: read_parquet, read_json, read_excel, read_sql
df = pd.read_parquet("data.parquet")

# Inspect
df.head(), df.tail(), df.sample(5)
df.shape, df.columns, df.dtypes
df.info()                                # memory + null counts
df.describe(include="all")               # summary stats
df.isna().sum()                          # nulls per column
df.nunique()                             # unique counts
df["col"].value_counts(normalize=True)   # category distribution
df.duplicated().sum()                    # duplicate rows
df.memory_usage(deep=True)

### Data Cleaning
# Missing values
df.dropna(subset=["col"])                # drop rows
df.fillna(0)                             # constant
df["col"].fillna(df["col"].median())     # impute
df.interpolate()                         # time-series gaps

# Duplicates / types
df.drop_duplicates(subset=["id"], keep="first")
df["col"].astype("category")
df["date"] = pd.to_datetime(df["date"])
df["num"] = pd.to_numeric(df["num"], errors="coerce")

# Strings
df["col"].str.lower().str.strip()
df["col"].str.replace(r"\D", "", regex=True)

# Outliers
q1, q3 = df["x"].quantile([0.25, 0.75]); iqr = q3 - q1
df = df[df["x"].between(q1 - 1.5*iqr, q3 + 1.5*iqr)]

# sklearn imputation
from sklearn.impute import SimpleImputer, KNNImputer
SimpleImputer(strategy="median").fit_transform(X)   # mean/median/most_frequent/constant
KNNImputer(n_neighbors=5).fit_transform(X)

### Feature Engineering
# Encoding (sklearn)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(X_cat)
OrdinalEncoder().fit_transform(X_cat)
LabelEncoder().fit_transform(y)          # target only

# Encoding (pandas)
pd.get_dummies(df, columns=["cat"], drop_first=True)

# Scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
StandardScaler().fit_transform(X)        # mean 0, std 1
MinMaxScaler().fit_transform(X)          # [0, 1]
RobustScaler().fit_transform(X)          # outlier-robust
PowerTransformer(method="yeo-johnson").fit_transform(X)

# Binning / polynomial / interactions
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures
KBinsDiscretizer(n_bins=5, encode="ordinal").fit_transform(X)
PolynomialFeatures(degree=2, interaction_only=True).fit_transform(X)

# Datetime features
df["year"], df["month"], df["dow"] = df["date"].dt.year, df["date"].dt.month, df["date"].dt.dayofweek
df["is_weekend"] = df["date"].dt.dayofweek >= 5

# Aggregations
df.groupby("user")["amt"].transform("mean")
df.groupby("user").agg(total=("amt", "sum"), cnt=("amt", "count"))

# Text features
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
TfidfVectorizer(max_features=5000, ngram_range=(1,2)).fit_transform(corpus)

# Feature selection
from sklearn.feature_selection import SelectKBest, f_classif, RFE, VarianceThreshold
SelectKBest(f_classif, k=10).fit_transform(X, y)
VarianceThreshold(threshold=0.0).fit_transform(X)

### Train/Test Split
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, TimeSeriesSplit

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
TimeSeriesSplit(n_splits=5)              # ordered/temporal data

# Imbalance (imbalanced-learn)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)

### Modeling
# Linear / classic
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

model = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)
model.fit(X_tr, y_tr)
preds = model.predict(X_te)
proba = model.predict_proba(X_te)[:, 1]

# Gradient boosting libraries
import xgboost as xgb
xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                  subsample=0.8, eval_metric="logloss").fit(X_tr, y_tr)

import lightgbm as lgb
lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31).fit(X_tr, y_tr)

from catboost import CatBoostClassifier
CatBoostClassifier(iterations=500, learning_rate=0.05, verbose=0).fit(X_tr, y_tr)

### Pipeline & ColumnTransformers
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

num_cols, cat_cols = ["age", "income"], ["city", "plan"]

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
pipe.fit(X_tr, y_tr)
pipe.predict(X_te)


### Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

grid = GridSearchCV(pipe, {"clf__C": [0.1, 1, 10]},
                    cv=5, scoring="f1", n_jobs=-1)
grid.fit(X_tr, y_tr)
grid.best_params_, grid.best_score_, grid.best_estimator_

# Bayesian / advanced (optuna)
import optuna
def objective(trial):
    C = trial.suggest_float("C", 1e-3, 1e2, log=True)
    return cross_val_score(LogisticRegression(C=C), X, y, cv=5).mean()
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
study.best_params

#### Evaluation
# Classification
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve,
    precision_recall_curve, log_loss)

accuracy_score(y_te, preds)
f1_score(y_te, preds, average="macro")
roc_auc_score(y_te, proba)
confusion_matrix(y_te, preds)
print(classification_report(y_te, preds))

# Regression
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
    r2_score, mean_absolute_percentage_error)
mean_squared_error(y_te, preds, squared=False)   # RMSE
r2_score(y_te, preds)

# Cross-validation scores
from sklearn.model_selection import cross_val_score, cross_validate
cross_val_score(model, X, y, cv=5, scoring="roc_auc")
cross_validate(model, X, y, cv=5, scoring=["f1", "precision"], return_train_score=True)


### Interpreability
# Built-in importances
model.feature_importances_               # tree models
model.coef_                              # linear models

# Permutation importance
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
permutation_importance(model, X_te, y_te, n_repeats=10)
PartialDependenceDisplay.from_estimator(model, X, ["age", "income"])

# SHAP
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_te)
shap.summary_plot(shap_values, X_te)

### Persistence & Deployment
import joblib
joblib.dump(pipe, "model.pkl")
pipe = joblib.load("model.pkl")

# Native formats
model.save_model("model.json")           # xgboost / lightgbm

# ONNX export
from skl2onnx import to_onnx
onx = to_onnx(pipe, X_tr[:1].astype(np.float32))

### Experiment Tracking
import mlflow
with mlflow.start_run():
    mlflow.log_params({"n_estimators": 300})
    mlflow.log_metrics({"f1": 0.87, "auc": 0.91})
    mlflow.sklearn.log_model(model, "model")

# Weights & Biases
import wandb
wandb.init(project="proj"); wandb.log({"loss": 0.1})

