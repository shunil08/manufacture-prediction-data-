
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# 1. Load the dataset
df = pd.read_csv("train_data.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst five rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

# 2. Separate inputs and target
target = "Defective"

if target not in df.columns:
    raise ValueError(
        f"Target '{target}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

X = df.drop(columns=[target])
y = df[target]

# Drop ID columns if present; they usually do not describe
# actual manufacturing conditions.
id_columns = [
    col for col in X.columns
    if col.lower() in [
        "id", "index", "unnamed: 0", "productid"
    ]
]
X = X.drop(columns=id_columns)

# 3. Identify numerical and categorical features
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 4. Build a machine-learning pipeline
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# 5. Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_val))

# 6. Train the model
print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# 7. Evaluate on data not used for training
predictions = model.predict(X_val)

print("\nValidation accuracy:",
      round(accuracy_score(y_val, predictions), 4))

print("\nClassification report:")
print(classification_report(y_val, predictions, zero_division=0))

print("\nConfusion matrix:")
print(confusion_matrix(y_val, predictions))

if y.nunique() == 2:
    probabilities = model.predict_proba(X_val)[:, 1]
    print("\nValidation ROC-AUC:",
          round(roc_auc_score(y_val, probabilities), 4))

# 8. Save the trained model
joblib.dump(model, "manufacturing_defect_model.joblib")

print("\nModel saved as manufacturing_defect_model.joblib")
