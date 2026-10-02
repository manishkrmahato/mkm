import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#from sklearn.datasets import fetch_california_housing
#import pandas as pd

#housing = fetch_california_housing(as_frame=True)

#df = housing.frame

#print(df.head())
#print(df.shape)

# ============================================================
# DATASET ACCESS
# ============================================================

FILE_PATH = "housing.csv"

df = pd.read_csv(FILE_PATH)

print("=" * 70)
print("CALIFORNIA HOUSING DATASET")
print("=" * 70)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset dimensions:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())


# ============================================================
# STANDARDIZE TARGET COLUMN NAME
# ============================================================

target_candidates = [
    "MedHouseVal",
    "median_house_value",
    "Median House Value",
    "median house value"
]

target = None

for column in target_candidates:
    if column in df.columns:
        target = column
        break

if target is None:
    print("\nAvailable columns:")
    print(df.columns.tolist())
    raise ValueError("Target column not found. Rename the target column to 'MedHouseVal' or 'median_house_value'.")

df = df.rename(columns={target: "MedHouseVal"})


# ============================================================
# REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates()

print("\nDataset dimensions after removing duplicates:")
print(df.shape)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

print("\nMissing values after treatment:")
print(df.isnull().sum())


# ============================================================
# IDENTIFY NUMERICAL PREDICTOR VARIABLES
# ============================================================

features = [column for column in df.columns if column != "MedHouseVal"]

numeric_features = df[features].select_dtypes(include=np.number).columns.tolist()

print("\nNumerical predictor variables:")
print(numeric_features)


# ============================================================
# OUTLIER ANALYSIS
# ============================================================

print("\nOutlier counts using IQR method:")

outlier_counts = {}

for column in numeric_features:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    count = ((df[column] < lower) | (df[column] > upper)).sum()
    outlier_counts[column] = count

print(pd.Series(outlier_counts))


# ============================================================
# TREAT OUTLIERS USING IQR CLIPPING
# ============================================================

for column in numeric_features:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[column] = df[column].clip(lower=lower, upper=upper)


print("\nOutliers treated using IQR clipping.")


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

correlation_matrix = df[numeric_features + ["MedHouseVal"]].corr()

print("\nCorrelation with target:")
print(correlation_matrix["MedHouseVal"].sort_values(ascending=False))


# ============================================================
# VISUALIZATION 1: HISTOGRAMS
# ============================================================

df[numeric_features].hist(
    figsize=(14, 10),
    bins=30
)

plt.suptitle("Distribution of Numerical Features")
plt.tight_layout()
plt.show()


# ============================================================
# VISUALIZATION 2: SCATTER PLOTS
# ============================================================

top_correlated_features = (
    correlation_matrix["MedHouseVal"]
    .drop("MedHouseVal")
    .abs()
    .sort_values(ascending=False)
    .head(3)
    .index
)

for column in top_correlated_features:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=df,
        x=column,
        y="MedHouseVal"
    )
    plt.title(f"{column} vs Median House Value")
    plt.xlabel(column)
    plt.ylabel("Median House Value")
    plt.show()


# ============================================================
# VISUALIZATION 3: BOX PLOTS
# ============================================================

plt.figure(figsize=(14, 7))

df[numeric_features].boxplot()

plt.title("Box Plot of Numerical Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================================
# VISUALIZATION 4: CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# ============================================================
# FEATURE SCALING AND TRAIN-TEST SPLIT
# ============================================================

X = df[numeric_features]
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

print("\nTraining set shape:", X_train_scaled.shape)
print("Testing set shape:", X_test_scaled.shape)


# ============================================================
# MODEL A
# THREE MOST HIGHLY CORRELATED FEATURES
# ============================================================

correlations = (
    df[numeric_features + ["MedHouseVal"]]
    .corr()["MedHouseVal"]
    .drop("MedHouseVal")
    .abs()
    .sort_values(ascending=False)
)

model_A_features = correlations.head(3).index.tolist()

print("\n" + "=" * 70)
print("MODEL A")
print("=" * 70)

print("Three most highly correlated features:")
print(model_A_features)

model_A = LinearRegression()

start_time = time.perf_counter()

model_A.fit(
    X_train_scaled[model_A_features],
    y_train
)

training_time_A = time.perf_counter() - start_time

prediction_A = model_A.predict(
    X_test_scaled[model_A_features]
)

mae_A = mean_absolute_error(y_test, prediction_A)
mse_A = mean_squared_error(y_test, prediction_A)
rmse_A = np.sqrt(mse_A)
r2_A = r2_score(y_test, prediction_A)


# ============================================================
# MODEL B
# FIVE SELECTED FEATURES
# ============================================================

model_B_features = correlations.head(5).index.tolist()

print("\n" + "=" * 70)
print("MODEL B")
print("=" * 70)

print("Five selected features:")
print(model_B_features)

model_B = LinearRegression()

start_time = time.perf_counter()

model_B.fit(
    X_train_scaled[model_B_features],
    y_train
)

training_time_B = time.perf_counter() - start_time

prediction_B = model_B.predict(
    X_test_scaled[model_B_features]
)

mae_B = mean_absolute_error(y_test, prediction_B)
mse_B = mean_squared_error(y_test, prediction_B)
rmse_B = np.sqrt(mse_B)
r2_B = r2_score(y_test, prediction_B)


# ============================================================
# MODEL C
# ALL AVAILABLE PREDICTOR VARIABLES
# ============================================================

model_C_features = numeric_features

print("\n" + "=" * 70)
print("MODEL C")
print("=" * 70)

print("All predictor features:")
print(model_C_features)

model_C = LinearRegression()

start_time = time.perf_counter()

model_C.fit(
    X_train_scaled[model_C_features],
    y_train
)

training_time_C = time.perf_counter() - start_time

prediction_C = model_C.predict(
    X_test_scaled[model_C_features]
)

mae_C = mean_absolute_error(y_test, prediction_C)
mse_C = mean_squared_error(y_test, prediction_C)
rmse_C = np.sqrt(mse_C)
r2_C = r2_score(y_test, prediction_C)


# ============================================================
# QUESTION 2: COMPARATIVE TABLE
# ============================================================

comparison = pd.DataFrame({
    "Model": ["Model A", "Model B", "Model C"],
    "Number of Features": [
        len(model_A_features),
        len(model_B_features),
        len(model_C_features)
    ],
    "Features": [
        ", ".join(model_A_features),
        ", ".join(model_B_features),
        ", ".join(model_C_features)
    ],
    "MAE": [
        mae_A,
        mae_B,
        mae_C
    ],
    "MSE": [
        mse_A,
        mse_B,
        mse_C
    ],
    "RMSE": [
        rmse_A,
        rmse_B,
        rmse_C
    ],
    "R2 Score": [
        r2_A,
        r2_B,
        r2_C
    ],
    "Training Time (seconds)": [
        training_time_A,
        training_time_B,
        training_time_C
    ]
})

print("\n" + "=" * 70)
print("QUESTION 2: MODEL COMPARISON")
print("=" * 70)

print(comparison.round(6).to_string(index=False))


# ============================================================
# IDENTIFY BEST MODEL
# ============================================================

best_model_index = comparison["R2 Score"].idxmax()

print("\nBest predictive model based on R2 Score:")

print(
    comparison.loc[
        best_model_index,
        ["Model", "Features", "R2 Score", "RMSE"]
    ]
)


# ============================================================
# QUESTION 2 TRADE-OFF
# ============================================================

print("\nTrade-off:")
print(
    "Models using fewer features generally require less computation, "
    "while models using more relevant features can improve prediction accuracy."
)

print(
    "Model C uses all available predictor variables and therefore provides "
    "the full feature information. Models A and B use fewer features and "
    "can provide a simpler model with lower feature complexity."
)


# ============================================================
# QUESTION 3: RFE FEATURE SELECTION
# ============================================================

print("\n" + "=" * 70)
print("QUESTION 3: RECURSIVE FEATURE ELIMINATION")
print("=" * 70)

rfe_estimator = LinearRegression()

number_of_features_to_select = min(5, len(numeric_features))

rfe = RFE(
    estimator=rfe_estimator,
    n_features_to_select=number_of_features_to_select
)

rfe.fit(X_train_scaled, y_train)

rfe_selected_features = X_train_scaled.columns[rfe.support_].tolist()

print("Features selected using RFE:")
print(rfe_selected_features)


# ============================================================
# RETRAIN MODEL USING RFE SELECTED FEATURES
# ============================================================

rfe_model = LinearRegression()

start_time = time.perf_counter()

rfe_model.fit(
    X_train_scaled[rfe_selected_features],
    y_train
)

rfe_training_time = time.perf_counter() - start_time

rfe_prediction = rfe_model.predict(
    X_test_scaled[rfe_selected_features]
)

rfe_mae = mean_absolute_error(y_test, rfe_prediction)
rfe_mse = mean_squared_error(y_test, rfe_prediction)
rfe_rmse = np.sqrt(rfe_mse)
rfe_r2 = r2_score(y_test, rfe_prediction)


# ============================================================
# MODEL C VS RFE MODEL
# ============================================================

feature_selection_comparison = pd.DataFrame({
    "Model": [
        "All Features Model",
        "RFE Selected Features Model"
    ],
    "Number of Features": [
        len(model_C_features),
        len(rfe_selected_features)
    ],
    "Features": [
        ", ".join(model_C_features),
        ", ".join(rfe_selected_features)
    ],
    "MAE": [
        mae_C,
        rfe_mae
    ],
    "MSE": [
        mse_C,
        rfe_mse
    ],
    "RMSE": [
        rmse_C,
        rfe_rmse
    ],
    "R2 Score": [
        r2_C,
        rfe_r2
    ],
    "Training Time (seconds)": [
        training_time_C,
        rfe_training_time
    ]
})

print("\n" + "=" * 70)
print("QUESTION 3: FEATURE SELECTION COMPARISON")
print("=" * 70)

print(
    feature_selection_comparison
    .round(6)
    .to_string(index=False)
)


# ============================================================
# QUESTION 3 CONCLUSION
# ============================================================

if rfe_r2 > r2_C:
    print("\nConclusion:")
    print(
        "The RFE selected feature model provides better predictive "
        "performance than the model using all features because it has "
        "a higher R2 Score."
    )

elif rfe_r2 < r2_C:
    print("\nConclusion:")
    print(
        "The model using all features provides better predictive "
        "performance than the RFE selected feature model because it "
        "has a higher R2 Score."
    )

else:
    print("\nConclusion:")
    print(
        "Both models provide the same R2 Score, so feature reduction "
        "maintains the predictive performance."
    )


# ============================================================
# SAVE RESULTS
# ============================================================

comparison.to_csv(
    "california_housing_model_comparison.csv",
    index=False
)

feature_selection_comparison.to_csv(
    "california_housing_feature_selection_comparison.csv",
    index=False
)

print("\nResult tables saved successfully.")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("Dataset shape:", df.shape)
print("Number of predictor variables:", len(numeric_features))

print("\nModel A features:")
print(model_A_features)

print("\nModel B features:")
print(model_B_features)

print("\nModel C features:")
print(model_C_features)

print("\nRFE selected features:")
print(rfe_selected_features)

print("\nModel comparison:")
print(comparison.round(4).to_string(index=False))

print("\nFeature-selection comparison:")
print(feature_selection_comparison.round(4).to_string(index=False))