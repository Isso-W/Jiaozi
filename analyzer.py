import pandas as pd


#====================================
# step 2: basic dataset information
#====================================

def analyze_dataset(df, name):

    print(f"\n========== {name} Dataset ==========")

    # informations of rows and columns
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())


#====================================
# step 3: columns analysis
#====================================

def analyze_feature_type(series):

    # numeric
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    
    # datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    
    # boolean
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    
    # object types (categorical or text)
    if series.dtype == "object":
        unique_count = series.nunique()

        if unique_count < 20:
            return "categorical"
        else:
            return "text"
        
    return "unknown"

# columns analysis missing ratio
def analyze_missing_ratio(series):
    ratio = series.isnull().mean() * 100
    return round(ratio, 2)

# columns analysis unique values
def count_unique_values(series):
    return int(series.nunique())

# analyze columns of each dataset
def analyze_columns(df):

    column_info_list = []

    for col in df.columns:

        series = df[col]

        column_info = {}

        # column name
        column_info["name"] = col

        #pandas dtype
        column_info["dtype"] = str(series.dtype)

        # feature type
        column_info["feature_type"] = analyze_feature_type(series)

        # missing ratio
        column_info["missing_ratio"] = analyze_missing_ratio(series)

        # unique values
        column_info["unique_values"] = count_unique_values(series)

        column_info_list.append(column_info)

    return column_info_list


# main function

def main():

    # step 1: load datasets

    titanic = pd.read_csv("data/titanic.csv")

    adult = pd.read_csv("data/adult.csv")

    house = pd.read_csv(
        "data/house_prices.csv",
        nrows = 10000
    )

    # step 2: basic info

    analyze_dataset(titanic, "Titanic")

    analyze_dataset(adult, "Adult")

    analyze_dataset(house, "House")

    # step 3: columns analysis

    print("\n===== Titanic Column Analysis =====")

    column_info = analyze_columns(titanic)

    # print column analysis results
    for col in column_info:
        print(col)

# python entry point
if __name__ == "__main__":
    main()