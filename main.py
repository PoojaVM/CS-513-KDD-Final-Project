# %% [markdown]
# # Course - CS-513 Knowledge Discovery and Data Mining

# %% [markdown]
# #### Problem Statement - User friendly guide to predict if a house in NYC would fall in a specific price range(L, M, H, VH)
# 
# #### Project Group 6
# 
# #### Team Members:
# - Akhil Vandanapu (20016200)
# - Anirudh Chintha (20016080)
# - Pooja Mule (20016077)

# %%
# Import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer

from sklearn.cluster import KMeans

import xgboost as xgb
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# %%
# Load dataset
df = pd.read_csv('nyc-rolling-sales.csv')

# Check dataset columns pre-cleanup
df.info()

# %% [markdown]
# ## Data Analysis and Preprocessing

# %%
# Clean dataset

print("Removing unwanted columns and replacing weird empty values with NaN")
print("\n")

# Drop Unnamed: 0 feature which is just a serial number
# Drop ADDRESS and LOT features as it is not required for analysis(We are going to use zip code instead)
df.drop(["ADDRESS", "Unnamed: 0", "LOT", "SALE DATE"], axis=1, inplace=True)

# Check and drop columns where all cells are empty or -
df = df.applymap(lambda x: pd.NA if str(x).strip() in ["-", ""] else x)
df.dropna(axis=1, how="all", inplace=True)

print("Dataset columns post-cleanup")
df.info()

# %%
# View dataset shape and head

print('(Rows, Columns):', df.shape)
df.head()

# %%
# Remove rows where commercial units and residential units do not add up to total uints
print(
    "Rows with total units != commercial units + residential units:",
    df[df["TOTAL UNITS"] != df["COMMERCIAL UNITS"] + df["RESIDENTIAL UNITS"]].shape[0],
)

df = df[df["TOTAL UNITS"] == (df["COMMERCIAL UNITS"] + df["RESIDENTIAL UNITS"])]

print(
    "Rows with total units != commercial units + residential units after removing:",
    df[df["TOTAL UNITS"] != df["COMMERCIAL UNITS"] + df["RESIDENTIAL UNITS"]].shape[0],
)

print("\n")

print("Data type of features", df.dtypes)

# %%
# Find categorical columns - columns with less than 10 unique values considered cateogrical for our purpose

categorical_columns = []

for column in df.columns:
    if df[column].dtype == 'object' or df[column].nunique() < 10:
        categorical_columns.append(column)


print('Categorical columns:')
print(categorical_columns, '\n')

# %%
# Filter out categorical features from 'categorical_columns' if they are truly useful in categorical sense
categorical_columns = [
    'BOROUGH',
    'NEIGHBORHOOD',
    'TAX CLASS AT PRESENT',
    'BUILDING CLASS CATEGORY',
    'BUILDING CLASS AT PRESENT',
    'TAX CLASS AT TIME OF SALE',
    'BUILDING CLASS AT TIME OF SALE',
]

for col in categorical_columns:
    df[col] = df[col].astype('category')
    df[col] = df[col].cat.codes

# Convert other feature data types to numeric wherever suitable
df['SALE PRICE'] = pd.to_numeric(df['SALE PRICE'])
df['LAND SQUARE FEET'] = pd.to_numeric(df['LAND SQUARE FEET'], errors='coerce')
df['GROSS SQUARE FEET']= pd.to_numeric(df['GROSS SQUARE FEET'], errors='coerce')

# %%
# Check if all features are appropriately set as per thier data types
print("Data type of features:")
df.dtypes

# %%
# We tried to remove extreme values from SALE PRICE but it resulted in loss of data. So we decided to keep them.

# %%
# Plot histogram for numerical data
for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        plt.figure(figsize=(8, 4))
        sns.histplot(df[column], kde=True)
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()

# %%
# Missing data - understand percentage of missing data

# Calculate missing % and create a barchart.
def show_missing_values(dataframe):
    missing_percentage = dataframe.isnull().mean() * 100

    plt.figure(figsize=(10, 6))
    missing_percentage.plot(kind="bar")
    plt.ylabel("Percentage of Missing Data")
    plt.xlabel("Features")
    plt.title("Percentage of Missing Data by Feature")
    plt.show()

show_missing_values(df)

# %% [markdown]
# ## Handle Missing Values

# %%
# Treating missing values

# Cateogorical values for SALE PRICE
SALE_PRICE_LABELS = ["Low", "Medium", "High", "Very High"]

# Replace YEAR BUILT = 0 with mode value
df["YEAR BUILT"] = df["YEAR BUILT"].replace(0, df["YEAR BUILT"].mode()[0])

# Remove rows with missing or 0 values in SALE PRICE which is target variable
df["SALE PRICE"] = df["SALE PRICE"].apply(lambda x: np.NAN if x <= 0 or "" else x)
df.dropna(subset=["SALE PRICE"], inplace=True)

# Verify if SALE PRICE has any NA values
print(
    "Number of null or 0 values after cleanup from SALE PRICE:",
    df["SALE PRICE"].isna().sum(),
)

# KNN imputation for 0 values of zip code
imputer = KNNImputer(n_neighbors=5)
df["ZIP CODE"] = imputer.fit_transform(df[["ZIP CODE"]])


# Delete the APARTMENT NUMBER since 77% of the values are missing.
df.drop("APARTMENT NUMBER", axis=1, inplace=True)


# Remove rows with missing values in TAX CLASS AT PRESENT and BUILDING CLASS AT PRESENT
df.dropna(subset=["TAX CLASS AT PRESENT", "BUILDING CLASS AT PRESENT"], inplace=True)

# Convert SALE PRICE to categorical variable
df["SALE PRICE"] = pd.qcut(df["SALE PRICE"], q=4, labels=SALE_PRICE_LABELS)

df.info()

# %%
print("Initial DF Shape: ", df.shape)
# Create duplicate df for imputation
df_median_impute = df.copy(deep=True)
df_mean_inpute = df.copy(deep=True)
df_knn_impute = df.copy(deep=True)
df_no_impute = df.copy(deep=True)

# Impute the missing values in LAND SQUARE FEET and GROSS SQUARE FEET using different methods

# Impute using median
df_median_impute['LAND SQUARE FEET'] = df_median_impute['LAND SQUARE FEET'].fillna(df_median_impute['LAND SQUARE FEET'].median())
df_median_impute['GROSS SQUARE FEET'] = df_median_impute['GROSS SQUARE FEET'].fillna(df_median_impute['GROSS SQUARE FEET'].median())
# # do log1p transformation to make the data more normally distributed
# df_median_impute['LAND SQUARE FEET'] = np.log1p(df_median_impute['LAND SQUARE FEET'])
# df_median_impute['GROSS SQUARE FEET'] = np.log1p(df_median_impute['GROSS SQUARE FEET'])
print("DF Median Impute Shape: ", df_median_impute.shape)

# Impute using mean
df_mean_inpute['LAND SQUARE FEET'] = df_mean_inpute['LAND SQUARE FEET'].fillna(df_mean_inpute['LAND SQUARE FEET'].mean())
df_mean_inpute['GROSS SQUARE FEET'] = df_mean_inpute['GROSS SQUARE FEET'].fillna(df_mean_inpute['GROSS SQUARE FEET'].mean())
# # do log1p transformation to make the data more normally distributed
# df_mean_inpute['LAND SQUARE FEET'] = np.log1p(df_mean_inpute['LAND SQUARE FEET'])
# df_mean_inpute['GROSS SQUARE FEET'] = np.log1p(df_mean_inpute['GROSS SQUARE FEET'])
print("DF Mean Impute Shape: ", df_mean_inpute.shape)

# Impute using KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_knn_impute['LAND SQUARE FEET'] = imputer.fit_transform(df_knn_impute[['LAND SQUARE FEET']])
df_knn_impute['GROSS SQUARE FEET'] = imputer.fit_transform(df_knn_impute[['GROSS SQUARE FEET']])
# # do log1p transformation to make the data more normally distributed
# df_knn_impute['LAND SQUARE FEET'] = np.log1p(df_knn_impute['LAND SQUARE FEET'])
# df_knn_impute['GROSS SQUARE FEET'] = np.log1p(df_knn_impute['GROSS SQUARE FEET'])
print("DF KNN Impute Shape: ", df_knn_impute.shape)

# Delete rows with missing values fir df_no_impute
df_no_impute.dropna(inplace=True)
# # do log1p transformation to make the data more normally distributed
# df_no_impute['LAND SQUARE FEET'] = np.log1p(df_no_impute['LAND SQUARE FEET'])
# df_no_impute['GROSS SQUARE FEET'] = np.log1p(df_no_impute['GROSS SQUARE FEET'])
print("DF No Impute Shape: ", df_no_impute.shape)

# %%
# Showing missing values after cleanup
show_missing_values(df_median_impute)
show_missing_values(df_mean_inpute)
show_missing_values(df_knn_impute)

# %% [markdown]
# ## Model Predictions

# %%
def plot_confusion_matrix(cm, title):
    print(title)
    print(cm)
    print("\n")


def get_predictions(model, attr_train, attr_test, target_train, target_test):
    model.fit(attr_train, target_train)
    return model.predict(attr_test)


# Return optimal k
def get_optimal_k(dataframe):
    attr = dataframe.drop("SALE PRICE", axis=1)
    target = dataframe["SALE PRICE"]

    scaler = MinMaxScaler()
    attr = scaler.fit_transform(attr)

    attr_train, attr_test, target_train, target_test = train_test_split(
        attr, target, test_size=0.3, random_state=42
    )
    k_values = range(1, 21)
    accuracy_map = dict()

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        target_pred = get_predictions(
            knn, attr_train, attr_test, target_train, target_test
        )
        accuracy_map[k] = accuracy_score(target_test, target_pred) * 100

    # Plot the accuracy for different values of k
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracy_map.values())
    plt.xticks(k_values)
    plt.xlabel("Value of k")
    plt.ylabel("Testing Accuracy")
    plt.title("Accuracy for different values of k")
    plt.show()

    # Get optimal k
    optimal_k = max(accuracy_map, key=accuracy_map.get)
    return optimal_k


# Return model based on type passed
def get_model(model_type, dataframe):
    if model_type == "gaussian_nb":
        return GaussianNB()
    elif model_type == "decision_tree":
        return DecisionTreeClassifier()
    elif model_type == "cart_5":
        return DecisionTreeClassifier(
            criterion="entropy", max_depth=5, splitter="best", max_leaf_nodes=5
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=7)
    elif model_type == "svm":
        return SVC()
    elif model_type == "knn":
        optimal_k = get_optimal_k(dataframe)
        print("The optimal number of neighbors is {}".format(optimal_k))
        return KNeighborsClassifier(n_neighbors=optimal_k)
    else:
        return None


# Write a common function to train and test the different models
def train_and_test_model(model_type, dataframe):
    # Get the model according to model_type passed
    model = get_model(model_type, dataframe)

    if model is None:
        print("Invalid model type")
        return 0

    attr = dataframe.drop("SALE PRICE", axis=1)
    target = dataframe["SALE PRICE"]

    if model_type == "svm":
        scaler = StandardScaler()
        attr = scaler.fit_transform(attr)
    elif model_type == "knn":
        scaler = MinMaxScaler()
        attr = scaler.fit_transform(attr)

    attr_train, attr_test, target_train, target_test = train_test_split(
        attr, target, test_size=0.3, random_state=42
    )

    target_pred = get_predictions(
        model, attr_train, attr_test, target_train, target_test
    )

    # Get the confusion matrix
    cm = confusion_matrix(target_test, target_pred)

    # Plot the confusion matrix
    plot_confusion_matrix(cm, "Confusion Matrix for " + model_type)

    # Print the classification report
    print(classification_report(target_test, target_pred))

    # Print the accuracy
    accuracy = accuracy_score(target_test, target_pred)
    print("Accuracy:", accuracy)

    return accuracy


def use_xg_boost_model(dataframe):
    label_encoder = LabelEncoder()

    df = dataframe.copy()
    df["SALE PRICE"] = label_encoder.fit_transform(df["SALE PRICE"])

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=["category"]).columns

    # Use pd.get_dummies() to create dummy variables for categorical columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    attr = df.drop("SALE PRICE", axis=1)
    target = df["SALE PRICE"]

    attr_train, attr_test, target_train, target_test = train_test_split(
        attr, target, test_size=0.3, random_state=42
    )

    # Define the model
    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier.fit(attr_train, target_train)
    y_pred = xgb_classifier.predict(attr_test)

    # Generate and print the confusion matrix
    cm = confusion_matrix(target_test, y_pred)
    plot_confusion_matrix(cm, "Confusion Matrix for XGBClassifier")

    # Print the classification report
    print(classification_report(target_test, y_pred))

    # Print the accuracy
    accuracy = accuracy_score(target_test, y_pred)
    print("Accuracy:", accuracy)

    return accuracy


def use_sequential_dense_modal(dataframe):
    # Get the feature and target columns
    attr = dataframe.drop("SALE PRICE", axis=1)
    target = dataframe["SALE PRICE"]

    # Convert the target column to categorical (one-hot encoding)
    encoded_target = to_categorical(target.cat.codes)

    # Scale the features
    scaler = StandardScaler()
    scaled_attr = scaler.fit_transform(attr)

    attr_train, attr_test, target_train, target_test = train_test_split(
        scaled_attr, encoded_target, test_size=0.3, random_state=42
    )

    n_features = attr_train.shape[1]

    # Build the neural network model
    model = Sequential()
    model.add(Dense(100, activation="softmax", input_shape=(n_features,)))
    model.add(Dense(100, activation="softmax"))
    model.add(Dense(4, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(attr_train, target_train, epochs=10, validation_split=0.3)

    loss, accuracy = model.evaluate(attr_test, target_test)
    print(f"Accuracy: {accuracy}")

    return accuracy

# %%
# Map each dataframe to a key and execute different models on it
# Save accuracies of every model for each dataframe in a dictionary

df_map = {
    "Median Impute": df_median_impute,
    "Mean Impute": df_mean_inpute,
    "KNN Impute": df_knn_impute,
    "No Impute": df_no_impute,
}

accuracy_map = dict()

for df_key in df_map.keys():
    print("Dataframe: ", df_key)
    print("\n")

    df_item = df_map[df_key]
    accuracy_map[df_key] = dict()

    print("KNN")
    accuracy_map[df_key]["knn"] = train_and_test_model("knn", df_item)
    print("\n")

    print("Gaussian NB")
    accuracy_map[df_key]["gaussian_nb"] = train_and_test_model("gaussian_nb", df_item)
    print("\n")

    print("Decision Tree")
    accuracy_map[df_key]["decision_tree"] = train_and_test_model("decision_tree", df_item)
    print("\n")

    print("CART 5")
    accuracy_map[df_key]["cart_5"] = train_and_test_model("cart_5", df_item)
    print("\n")

    print("Random Forest")
    accuracy_map[df_key]["random_forest"] = train_and_test_model("random_forest", df_item)
    print("\n")

    print("SVM")
    accuracy_map[df_key]["svm"] = train_and_test_model("svm", df_item)
    print("\n")

    print("XG Boost")
    accuracy_map[df_key]["xg_boost"] = use_xg_boost_model(df_item)
    print("\n")

    print("Sequential Dense Model")
    accuracy_map[df_key]["ann"] = use_sequential_dense_modal(df_item)
    print("\n\n")

# %%
# Using the accuracies map and plotting a bar chart to compare the accuracies of different models acros different imputation methods

models = ["knn", "gaussian_nb", "decision_tree", "cart_5", "random_forest", "svm", "xg_boost", "ann"]

dfs = df_map.keys()
num_groups = len(models)
r = np.arange(num_groups)

bar_width = 0.1

plt.figure(figsize=(15, 8))
for i, df_name in enumerate(dfs):
    accuracies = [accuracy_map[df_name].get(model, 0) for model in models]
    plt.bar(r + i * bar_width, accuracies, color=plt.cm.Paired(i / len(dfs)), width=bar_width, edgecolor='gray', label=df_name)

plt.xlabel('Model', fontweight='bold')
plt.xticks([r + bar_width * (len(dfs) / 2) for r in range(num_groups)], models)

plt.legend(title="Imputation Methods", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Across Different Imputation Methods')
plt.ylim(0, 1)

plt.show()

# %% [markdown]
# ## Finding the best model, best dataset and calculating its accuracy

# %%
# Get the model with highest accuracy
max_accuracy = 0
max_accuracy_model = None
max_accuracy_df = None

for df_key in accuracy_map.keys():
    for model in accuracy_map[df_key].keys():
        if accuracy_map[df_key][model] > max_accuracy:
            max_accuracy = accuracy_map[df_key][model]
            max_accuracy_model = model
            max_accuracy_df = df_key

print("Max accuracy:", max_accuracy)
print("Max accuracy model:", max_accuracy_model)
print("Max accuracy df:", max_accuracy_df)

# %%
# Calculate the accuracy one last time
if max_accuracy_model == 'xg_boost':
    use_xg_boost_model(df_map[max_accuracy_df])
elif max_accuracy_model == 'ann':
    use_sequential_dense_modal(df_map[max_accuracy_df])
else:
    train_and_test_model(max_accuracy_model, df_map[max_accuracy_df])

# %%
# Get the model with lowest accuracy
lowest_accuracy = 100
lowest_accuracy_model = None
lowest_accuracy_df = None

for df_key in accuracy_map.keys():
    for model in accuracy_map[df_key].keys():
        if accuracy_map[df_key][model] < lowest_accuracy:
            lowest_accuracy = accuracy_map[df_key][model]
            lowest_accuracy_model = model
            lowest_accuracy_df = df_key

print("Lowest accuracy:", lowest_accuracy)
print("Lowest accuracy model:", lowest_accuracy_model)
print("Lowest accuracy df:", lowest_accuracy_df)


