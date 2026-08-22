# ============================================================
# TASK 3: CAR PRICE PREDICTION WITH MACHINE LEARNING
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------

st.title("🚗 Car Price Prediction with Machine Learning")

st.write(
    "Upload a car dataset and train a Machine Learning regression "
    "model to predict car prices based on features such as brand, "
    "horsepower, mileage, engine size, and other available attributes."
)

st.divider()

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

st.sidebar.header("⚙️ Model Settings")

test_size = st.sidebar.slider(
    "Test Data Percentage",
    min_value=10,
    max_value=40,
    value=20,
    step=5
)

n_estimators = st.sidebar.slider(
    "Number of Trees",
    min_value=50,
    max_value=300,
    value=100,
    step=50
)

random_state = st.sidebar.number_input(
    "Random State",
    min_value=0,
    max_value=100,
    value=42
)

# ------------------------------------------------------------
# DATASET UPLOAD
# ------------------------------------------------------------

st.header("📂 Upload Car Dataset")

uploaded_file = st.file_uploader(
    "Upload your CSV or Excel car dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is None:

    st.info(
        "👆 Upload the car dataset provided for Task 3 to begin."
    )

    st.stop()

# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------

try:

    if uploaded_file.name.lower().endswith(".csv"):

        df = pd.read_csv(uploaded_file)

    else:

        df = pd.read_excel(uploaded_file)

except Exception as e:

    st.error(f"Error loading dataset: {e}")
    st.stop()

# ------------------------------------------------------------
# CHECK DATASET
# ------------------------------------------------------------

if df.empty:

    st.error("The uploaded dataset is empty.")
    st.stop()

st.success("✅ Dataset loaded successfully!")

# ------------------------------------------------------------
# CLEAN COLUMN NAMES
# ------------------------------------------------------------

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace("-", "_", regex=False)
)

# ------------------------------------------------------------
# DATASET OVERVIEW
# ------------------------------------------------------------

st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

with col4:
    st.metric(
        "Duplicate Rows",
        int(df.duplicated().sum())
    )

st.subheader("🔍 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)

st.subheader("📋 Available Columns")

st.write(list(df.columns))

# ------------------------------------------------------------
# REMOVE DUPLICATES
# ------------------------------------------------------------

duplicate_count = int(df.duplicated().sum())

if duplicate_count > 0:

    df = df.drop_duplicates()

    st.success(
        f"Removed {duplicate_count} duplicate rows."
    )

# ------------------------------------------------------------
# FIND PRICE / TARGET COLUMN
# ------------------------------------------------------------

target_column = None

possible_target_columns = [
    "price",
    "selling_price",
    "sellingprice",
    "car_price",
    "carprice",
    "present_price",
    "presentprice",
    "msrp",
    "sale_price",
    "resale_price"
]

for column in possible_target_columns:

    if column in df.columns:

        target_column = column
        break

# Search for columns containing price

if target_column is None:

    for column in df.columns:

        if "price" in column:

            target_column = column
            break

# ------------------------------------------------------------
# MANUAL TARGET SELECTION IF NOT FOUND
# ------------------------------------------------------------

if target_column is None:

    st.warning(
        "The price column could not be detected automatically."
    )

    target_column = st.selectbox(
        "Select the column containing the car price:",
        df.columns
    )

st.success(
    f"🎯 Target column selected: **{target_column}**"
)

# ------------------------------------------------------------
# CONVERT TARGET TO NUMERIC
# ------------------------------------------------------------

df[target_column] = (
    df[target_column]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("$", "", regex=False)
    .str.replace("₹", "", regex=False)
    .str.strip()
)

df[target_column] = pd.to_numeric(
    df[target_column],
    errors="coerce"
)

# Remove rows without price

df = df.dropna(
    subset=[target_column]
)

# ------------------------------------------------------------
# REMOVE INVALID TARGET VALUES
# ------------------------------------------------------------

df = df[
    df[target_column] > 0
]

if len(df) < 10:

    st.error(
        "The dataset has too few valid price records for training."
    )

    st.stop()

# ------------------------------------------------------------
# PREPARE FEATURES AND TARGET
# ------------------------------------------------------------

X = df.drop(
    columns=[target_column]
)

y = df[target_column]

# ------------------------------------------------------------
# REMOVE COLUMNS THAT ARE NOT USEFUL
# ------------------------------------------------------------

# Remove completely empty columns

empty_columns = [
    column
    for column in X.columns
    if X[column].isnull().all()
]

if empty_columns:

    X = X.drop(
        columns=empty_columns
    )

# Remove columns with unique values for every row
# when they look like IDs

id_columns = []

for column in X.columns:

    column_name = column.lower()

    if (
        ("id" in column_name or "index" in column_name)
        and X[column].nunique() == len(X)
    ):

        id_columns.append(column)

if id_columns:

    X = X.drop(
        columns=id_columns
    )

# ------------------------------------------------------------
# IDENTIFY NUMERIC AND CATEGORICAL FEATURES
# ------------------------------------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# ------------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)

# ------------------------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size / 100,
    random_state=int(random_state)
)

# ------------------------------------------------------------
# MACHINE LEARNING MODEL
# ------------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=int(n_estimators),
    random_state=int(random_state),
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)

# ------------------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------------------

with st.spinner("🤖 Training Machine Learning model..."):

    pipeline.fit(
        X_train,
        y_train
    )

st.success("✅ Model trained successfully!")

# ------------------------------------------------------------
# PREDICTIONS
# ------------------------------------------------------------

y_pred = pipeline.predict(
    X_test
)

# ------------------------------------------------------------
# MODEL EVALUATION
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)

# ------------------------------------------------------------
# PERFORMANCE
# ------------------------------------------------------------

st.divider()

st.header("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "MAE",
        f"{mae:,.2f}"
    )

with col2:

    st.metric(
        "RMSE",
        f"{rmse:,.2f}"
    )

with col3:

    st.metric(
        "R² Score",
        f"{r2:.3f}"
    )

with col4:

    st.metric(
        "Training Records",
        len(X_train)
    )

st.write(
    """
    **MAE:** Mean Absolute Error  
    **RMSE:** Root Mean Squared Error  
    **R² Score:** Measures how well the model explains price variation.
    """
)

# ------------------------------------------------------------
# ACTUAL VS PREDICTED
# ------------------------------------------------------------

st.header("📊 Actual vs Predicted Prices")

comparison_df = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

st.dataframe(
    comparison_df.head(20).round(2),
    use_container_width=True
)

# ------------------------------------------------------------
# SCATTER PLOT
# ------------------------------------------------------------

fig, ax = plt.subplots()

ax.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
)

ax.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

ax.set_title(
    "Actual vs Predicted Car Prices"
)

ax.set_xlabel(
    "Actual Price"
)

ax.set_ylabel(
    "Predicted Price"
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ------------------------------------------------------------
# PREDICTION SECTION
# ------------------------------------------------------------

st.divider()

st.header("🚗 Predict Car Price")

st.write(
    "Enter the car details below. The available input fields "
    "are generated automatically from your uploaded dataset."
)

prediction_input = {}

# ------------------------------------------------------------
# NUMERIC INPUTS
# ------------------------------------------------------------

for column in numeric_features:

    median_value = X[column].median()

    if pd.isna(median_value):

        median_value = 0.0

    prediction_input[column] = st.number_input(
        column.replace("_", " ").title(),
        value=float(median_value)
    )

# ------------------------------------------------------------
# CATEGORICAL INPUTS
# ------------------------------------------------------------

for column in categorical_features:

    values = (
        X[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    values = sorted(values)

    if len(values) > 0:

        prediction_input[column] = st.selectbox(
            column.replace("_", " ").title(),
            values
        )

    else:

        prediction_input[column] = ""

# ------------------------------------------------------------
# PREDICT BUTTON
# ------------------------------------------------------------

if st.button(
    "🔮 Predict Car Price",
    use_container_width=True
):

    input_df = pd.DataFrame(
        [prediction_input]
    )

    predicted_price = pipeline.predict(
        input_df
    )[0]

    st.success(
        f"🚗 Estimated Car Price: **{predicted_price:,.2f}**"
    )

# ------------------------------------------------------------
# FEATURE INFORMATION
# ------------------------------------------------------------

st.divider()

st.header("🧠 Features Used by the Model")

col1, col2 = st.columns(2)

with col1:

    st.subheader("🔢 Numerical Features")

    if numeric_features:

        for feature in numeric_features:

            st.write(
                f"• {feature.replace('_', ' ').title()}"
            )

    else:

        st.write("No numerical features found.")

with col2:

    st.subheader("🔤 Categorical Features")

    if categorical_features:

        for feature in categorical_features:

            st.write(
                f"• {feature.replace('_', ' ').title()}"
            )

    else:

        st.write("No categorical features found.")

# ------------------------------------------------------------
# DOWNLOAD PREDICTIONS
# ------------------------------------------------------------

comparison_df["Error"] = (
    comparison_df["Actual Price"]
    - comparison_df["Predicted Price"]
)

csv_data = comparison_df.to_csv(
    index=False
)

st.download_button(
    label="⬇️ Download Predictions",
    data=csv_data,
    file_name="car_price_predictions.csv",
    mime="text/csv"
)

# ------------------------------------------------------------
# KEY INSIGHTS
# ------------------------------------------------------------

st.header("💡 Key Insights")

st.write(
    f"""
    - The model used **Random Forest Regression**.
    - Training records: **{len(X_train)}**
    - Testing records: **{len(X_test)}**
    - Mean Absolute Error: **{mae:,.2f}**
    - Root Mean Squared Error: **{rmse:,.2f}**
    - R² Score: **{r2:.3f}**
    - The model can predict car prices using the features
      available in the uploaded dataset.
    """
)

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.divider()

st.caption(
    "Task 3: Car Price Prediction with Machine Learning | Internship Project"
)