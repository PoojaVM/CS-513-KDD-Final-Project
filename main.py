# %% [markdown]
# ### Course - CS-513 Knowledge Discovery and Data Mining
# #### Problem Statement - Predict the prices of real estate in New York City using the dataset from Kaggle

# %%
# Import required libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Load dataset
df = pd.read_csv('nyc-rolling-sales.csv')

# Dataset columns pre-cleanup
df.info()

# %%
# Clean dataset

# Drop Unnamed: 0 feature which is just a serial number
# Drop ADDRESS and LOT features as it is not required for analysis.
df.drop(["ADDRESS", "Unnamed: 0", "LOT", "SALE DATE"], axis=1, inplace=True)

# Check and drop columns where all cells are empty or -
df = df.applymap(lambda x: pd.NA if str(x).strip() in ['-', ''] else x)
df.dropna(axis=1, how='all', inplace=True)

# Dataset columns post-cleanup
df.info()

# %%
# View dataset statistics

print('(Rows, Columns):', df.shape)
df.head()


# %%
# Remove rows with total units is not equal to commercial units + residential units
df = df[df["TOTAL UNITS"] == (df["COMMERCIAL UNITS"] + df["RESIDENTIAL UNITS"])]

print(
    "Rows with total units != commercial units + residential units:",
    df[df["TOTAL UNITS"] != df["COMMERCIAL UNITS"] + df["RESIDENTIAL UNITS"]].shape[0],
)

print("\n")

# Check data type of features
print("Data type of features", df.dtypes)

# %%
# Find categorical columns - columns with less than 10 unique values considered cateogrical for the purpose

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
    # Convert to categorical data type
    df[col] = df[col].astype('category')
    # Do label encoding
    df[col] = df[col].cat.codes

# Convert other feature data types to numeric wherever suitable
df['SALE PRICE'] = pd.to_numeric(df['SALE PRICE'])
df['LAND SQUARE FEET'] = pd.to_numeric(df['LAND SQUARE FEET'], errors='coerce')
df['GROSS SQUARE FEET']= pd.to_numeric(df['GROSS SQUARE FEET'], errors='coerce')

# # Convert SALE DATE to datetime and extact year
# df['SALE DATE'] = pd.to_datetime(df['SALE DATE'], errors='coerce')
# # df['SALE YEAR'] = df['SALE DATE'].dt.year
# df["SALE DATE"] = pd.DatetimeIndex(df["SALE DATE"]).year

# Replace YEAR BUILT = 0 with mode value
df['YEAR BUILT'] = df['YEAR BUILT'].replace(0, df['YEAR BUILT'].mode()[0])

# Find number of SALE PRICE columns with 0 value
print('Number of SALE PRICE columns with 0 value:', df[df['SALE PRICE'] == 0].shape[0])

# Remove rows with SALE PRICE = 0
df = df[df['SALE PRICE'] != 0]

# Rows with SALE PRICE = 0 after cleanup
print('Number of SALE PRICE columns with 0 value after cleanup:', df[df['SALE PRICE'] == 0].shape[0])


# %%
# Check if all features are appropriately set as per thier data types
print("Data type of features:")
df.dtypes

# %%

# We tried to remove extreme values from SALE PRICE but it resulted in loss of data. So we decided to keep them.



# %%
# # Plot histogram for numerical data
# for column in df.columns:
#     # Check if the column is numeric
#     if pd.api.types.is_numeric_dtype(df[column]):
#         # Plot a histogram for numeric data
#         plt.figure(figsize=(8, 4))
#         sns.histplot(df[column], kde=True)
#         plt.title(f"Histogram of {column}")
#         plt.xlabel(column)
#         plt.ylabel("Frequency")
#         plt.show()

    # # Check if the column is categorical
    # elif pd.api.types.is_categorical_dtype(df[column]):
    #     # Plot a countplot for categorical data
    #     plt.figure(figsize=(8, 4))
    #     sns.countplot(x=column, data=df)
    #     plt.title(f'Countplot of {column}')
    #     plt.xlabel(column)
    #     plt.ylabel('Count')
    #     plt.xticks(rotation=45)
    #     plt.show()

# %%
# Missing data - understand percentage of missing data


def show_missing_values(dataframe):
    # Calculate the percentage of missing data in each column
    missing_percentage = dataframe.isnull().mean() * 100

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    missing_percentage.plot(kind="bar")
    plt.ylabel("Percentage of Missing Data")
    plt.xlabel("Features")
    plt.title("Percentage of Missing Data by Feature")

    # Show the plot
    plt.show()

show_missing_values(df)

# %%
df.dtypes

# get nunique values of zip code
df['ZIP CODE'].nunique()

# %% [markdown]
# ## Treating Missing Values

# %%
# Treating missing values

# Cateogorical values for SALE PRICE
SALE_PRICE_LABELS = ["Low", "Medium", "High", "Very High"]

# Remove rows with missing or 0 values in SALE PRICE which is target variable
df["SALE PRICE"] = df["SALE PRICE"].apply(lambda x: np.NAN if x <= 0 or "" else x)
df.dropna(subset=["SALE PRICE"], inplace=True)

# Check if SALE PRICE has any NA values
print(
    "Number of null or 0 values after cleanup from SALE PRICE:",
    df["SALE PRICE"].isna().sum(),
)


# Delete the APARTMENT NUMBER since 77% of the values are missing and it is not a useful feature
df.drop("APARTMENT NUMBER", axis=1, inplace=True)


# Remove rows with missing values in TAX CLASS AT PRESENT and BUILDING CLASS AT PRESENT
df.dropna(subset=["TAX CLASS AT PRESENT", "BUILDING CLASS AT PRESENT"], inplace=True)

# Do k-means clustering to remove outliers from all numeric features
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
def kmeans_remove_outliers(df, n_clusters=5, random_state=0):
    # Standardize selected features
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(df)
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(standardized_features)
    # Calculate distances of each data point from the cluster centers
    distances = kmeans.transform(standardized_features)
    # Find closest cluster for each data point
    closest_cluster_distances = np.min(distances, axis=1)
    # Determine the threshold value for outliers
    threshold_distance = np.mean(closest_cluster_distances) + 3 * np.std(closest_cluster_distances)
    # Flag the outliers
    outliers = closest_cluster_distances > threshold_distance
    # Remove the outliers
    df = df[~outliers]
    # Change SALE PRICE to categorical variable
    df["SALE PRICE"] = pd.qcut(df["SALE PRICE"], q=4, labels=SALE_PRICE_LABELS)
    return df


df["SALE PRICE"].describe()

# change SALE PRICE to categorical variable
# SALE_PRICE_LABELS = ["Low", "Medium", "High", "Very High"]
# df["SALE PRICE"] = pd.qcut(df["SALE PRICE"], q=4, labels=SALE_PRICE_LABELS)

df.info()

# %%
# Check if LAND SQUARE FEET and GROSS SQUARE FEET are normally distributed

# plt.figure(figsize=(8, 4))
# sns.histplot(df['LAND SQUARE FEET'], kde=True)
# plt.title('Histogram of LAND SQUARE FEET')
# plt.xlabel('LAND SQUARE FEET')
# plt.ylabel('Frequency')
# plt.show()

# plt.figure(figsize=(8, 4))
# sns.histplot(df['GROSS SQUARE FEET'], kde=True)
# plt.title('Histogram of GROSS SQUARE FEET')
# plt.xlabel('GROSS SQUARE FEET')
# plt.ylabel('Frequency')
# plt.show()

# %%
# Create duplicate df for imputation
df_median_impute = df.copy()
df_mean_inpute = df.copy()
df_knn_impute = df.copy()
df_no_impute = df.copy()

# Impute the missing values in LAND SQUARE FEET and GROSS SQUARE FEET using different methods

# Impute using median
df_median_impute['LAND SQUARE FEET'] = df_median_impute['LAND SQUARE FEET'].fillna(df_median_impute['LAND SQUARE FEET'].median())
df_median_impute['GROSS SQUARE FEET'] = df_median_impute['GROSS SQUARE FEET'].fillna(df_median_impute['GROSS SQUARE FEET'].median())
# # do log1p transformation to make the data more normally distributed
# df_median_impute['LAND SQUARE FEET'] = np.log1p(df_median_impute['LAND SQUARE FEET'])
# df_median_impute['GROSS SQUARE FEET'] = np.log1p(df_median_impute['GROSS SQUARE FEET'])

df_median_impute = kmeans_remove_outliers(df_median_impute)

# Impute using mean
df_mean_inpute['LAND SQUARE FEET'] = df_mean_inpute['LAND SQUARE FEET'].fillna(df_mean_inpute['LAND SQUARE FEET'].mean())
df_mean_inpute['GROSS SQUARE FEET'] = df_mean_inpute['GROSS SQUARE FEET'].fillna(df_mean_inpute['GROSS SQUARE FEET'].mean())
# # do log1p transformation to make the data more normally distributed
# df_mean_inpute['LAND SQUARE FEET'] = np.log1p(df_mean_inpute['LAND SQUARE FEET'])
# df_mean_inpute['GROSS SQUARE FEET'] = np.log1p(df_mean_inpute['GROSS SQUARE FEET'])
df_mean_inpute = kmeans_remove_outliers(df_mean_inpute)

# Impute using KNN
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_knn_impute['LAND SQUARE FEET'] = imputer.fit_transform(df_knn_impute[['LAND SQUARE FEET']])
df_knn_impute['GROSS SQUARE FEET'] = imputer.fit_transform(df_knn_impute[['GROSS SQUARE FEET']])
# # do log1p transformation to make the data more normally distributed
# df_knn_impute['LAND SQUARE FEET'] = np.log1p(df_knn_impute['LAND SQUARE FEET'])
# df_knn_impute['GROSS SQUARE FEET'] = np.log1p(df_knn_impute['GROSS SQUARE FEET'])
df_knn_impute = kmeans_remove_outliers(df_knn_impute)

# Delete rows with missing values fir df_no_impute
df_no_impute.dropna(inplace=True)
# # do log1p transformation to make the data more normally distributed
# df_no_impute['LAND SQUARE FEET'] = np.log1p(df_no_impute['LAND SQUARE FEET'])
# df_no_impute['GROSS SQUARE FEET'] = np.log1p(df_no_impute['GROSS SQUARE FEET'])
df_no_impute = kmeans_remove_outliers(df_no_impute)

# %%
# Showing missing values after cleanup
# show_missing_values(df_median_impute)
# show_missing_values(df_mean_inpute)
# show_missing_values(df_knn_impute)

# %%
# Model building
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical



def plot_confusion_matrix(cm, title):
    print(title)
    print(cm)
    # ax = plt.subplot()
    # sns.heatmap(cm, annot=True, fmt="g", ax=ax)
    # # annot=True to annotate cells, ftm='g' to disable scientific notation

    # # labels, title and ticks
    # ax.set_xlabel("Predicted labels")
    # ax.set_ylabel("True labels")
    # ax.set_title(title)
    # ax.xaxis.set_ticklabels(SALE_PRICE_LABELS)
    # ax.yaxis.set_ticklabels(SALE_PRICE_LABELS)


def get_predictions(model, X_train, X_test, y_train, y_test):
    # Fit the model to the training set
    model.fit(X_train, y_train)

    # Make predictions on the test set
    return model.predict(X_test)

# Return optimal k
def get_optimal_k(dataframe):
    # Get the feature and target columns
    X = dataframe.drop("SALE PRICE", axis=1)
    y = dataframe["SALE PRICE"]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    k_values = range(1, 21)
    accuracy_map = dict()

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        target_pred = get_predictions(knn, X_train, X_test, y_train, y_test)
        accuracy_map[k] = accuracy_score(y_test, target_pred) * 100

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
        return DecisionTreeClassifier(max_depth=5)
    elif model_type == "random_forest":
        return RandomForestClassifier()
    elif model_type == "svm":
        return SVC()
    elif model_type == "logistic_regression":
        return LogisticRegression()
    elif model_type == "knn":
        # Note for testing - Uncomment below code to get optimal k. It has been commented to avoid running it everytime.
        # optimal_k = get_optimal_k(dataframe)
        # Above function returns optimal k = 8
        optimal_k = 8
        print("The optimal number of neighbors is {}".format(optimal_k))
        return KNeighborsClassifier(n_neighbors=optimal_k)
    # elif model_type == "xg_boost":
    #     return xgb.XGBClassifier()
    # elif model_type == "sequential_dense":
    #     return Sequential()
    else:
        return None

# Write a common function to train and test the different models
def train_and_test_model(model_type, dataframe):
    # Get the model
    model = get_model(model_type, dataframe)

    # Get the feature and target columns
    X = dataframe.drop("SALE PRICE", axis=1)
    y = dataframe["SALE PRICE"]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Get predictions
    target_pred = get_predictions(model, X_train, X_test, y_train, y_test)

    # Get the confusion matrix
    cm = confusion_matrix(y_test, target_pred)

    # Plot the confusion matrix
    plot_confusion_matrix(cm, "Confusion Matrix for " + model_type)

    # Print the classification report
    print(classification_report(y_test, target_pred))

    # Print the accuracy
    print("Accuracy:", accuracy_score(y_test, target_pred))

def use_xg_boost_model(dataframe):
    # Initialize the label encoder for the target variable
    label_encoder = LabelEncoder()

    # Copy the dataframe to avoid modifying the original data
    df = dataframe.copy()

    # Fit and transform the 'SALE PRICE' column
    df["SALE PRICE"] = label_encoder.fit_transform(df["SALE PRICE"])

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=["category"]).columns

    # Use pd.get_dummies() to create dummy variables for categorical columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Split the data into features and target
    X = df.drop("SALE PRICE", axis=1)
    y = df["SALE PRICE"]

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Create a XGBClassifier model
    xgb_classifier = xgb.XGBClassifier()

    # Fit the model to the training set
    xgb_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = xgb_classifier.predict(X_test)

    # Generate and print the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, "Confusion Matrix for XGBClassifier")

    # Print the classification report
    print(classification_report(y_test, y_pred))

    # Print the accuracy
    print("Accuracy:", accuracy_score(y_test, y_pred))


def use_sequential_dense_modal(dataframe):

    # Get the feature and target columns
    X = dataframe.drop("SALE PRICE", axis=1)
    y = dataframe["SALE PRICE"]

    # Convert the target column to categorical (one-hot encoding)
    y_encoded = to_categorical(y.cat.codes)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.3, random_state=42
    )

    # Get the number of input features
    n_features = X_train.shape[1]

    # Create a Sequential model
    model = Sequential()
    model.add(Dense(100, activation="relu", input_shape=(n_features,)))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(4, activation="softmax"))  # Output layer for 4-class classification

    # Compile the model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Fit the model to the training data
    model.fit(X_train, y_train, epochs=10, validation_split=0.3)

    # Evaluating the Model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy}")

# %%
df_map = {
    "Median Impute": df_median_impute,
    "Mean Impute": df_mean_inpute,
    "KNN Impute": df_knn_impute,
    "No Impute": df_no_impute,
}

for df_key in df_map.keys():
    # Get df_name from key
    print("Dataframe: ", df_key)
    print("\n")

    df_item = df_map[df_key]

    print("KNN")
    train_and_test_model("knn", df_item)
    print("\n")

    print("Gaussian NB")
    train_and_test_model("gaussian_nb", df_item)
    print("\n")

    print("Decision Tree")
    train_and_test_model("decision_tree", df_item)
    print("\n")

    print("CART 5")
    train_and_test_model("cart_5", df_item)
    print("\n")

    print("Random Forest")
    train_and_test_model("random_forest", df_item)
    print("\n")

    print("SVM")
    train_and_test_model("svm", df_item)
    print("\n")


    print("Logistic Regression")
    train_and_test_model("logistic_regression", df_item)
    print("\n")

    print("XG Boost")
    use_xg_boost_model(df_item)
    print("\n")

    print("Sequential Dense Model")
    use_sequential_dense_modal(df_item)
    print("\n\n")


