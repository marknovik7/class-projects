import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

"""
Mark Grivnin
CS 677 - Project
Classification of credit scores.
"""


def drop_columns(df, d_cols):
    """
    :param df: data frame to read
    :param d_cols: columns to drop
    :return: data frame with dropped columns and nan values
    """
    return df.drop(d_cols, axis=1).dropna()


def to_float(df, cols):
    """
    :param df: data frame to read
    :param cols: columns to covert to floats
    :return: converted data frame
    """
    df[cols] = df[cols].astype(float)
    return df


def clean_column(df, column):
    """
    :param df: data frame to read
    :param column: column to clean
    :return: cleaned data frame
    """
    df[column] = df[column].apply(lambda x: re.sub(r'[_%!@#{}[\]()>+$:;]', '', str(x)))
    df = df[df[column] != '']
    return df


def cat_to_num(df, column):
    """
    :param df: data frame to read
    :param column: column to parse through
    :return: data frame with converted column
    """
    replace_dict = {}

    for val in df[column].unique():
        if str(val)[0].isalpha():
            replace_dict[val] = len(replace_dict) + 1
        else:
            df = df[df[column] != val]

    df = df.replace(replace_dict)
    return df


def drop_outlier(df, col_name):
    """
    :param df: data frame to read
    :param col_name: column to drop outliers from
    :return: data frame without the outliers
    """
    q1, q3 = df[col_name].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr
    return df[(df[col_name] >= lower_bound) & (df[col_name] <= upper_bound)]


def plot_data(df):
    """
    :param df: data frame to use for visualization
    :return: nothing
    """
    sns.set(rc={'figure.figsize': (12, 8)})
    sns.barplot(x=df['Occupation'].value_counts().index, y=df['Occupation'].value_counts().values)
    plt.title('Bar Graph for Occupation', fontsize=17)
    plt.xlabel('Occupation', fontsize=13)
    plt.ylabel('Count', fontsize=13)
    plt.xticks(rotation=40)
    plt.show()

    # heatmap
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True)
    plt.title('Heatmap for Credit Related Variables', fontsize=17)
    plt.xticks(rotation=25)
    plt.show()


def visualize_k(k_values, accuracies):
    """
    :param k_values: neighbor values
    :param accuracies: corresponding accuracies to k values
    :return: nothing
    """
    plt.plot(k_values, accuracies)
    plt.title("KNN plot of Accuracy")
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.show()


def split_data(df, train_columns, test_columns):
    """
    :param df: data frame to read
    :param train_columns: training columns
    :param test_columns: testing columns
    :return: split training and testing data sets
    """
    x_train, x_test, y_train, y_test = train_test_split(df[train_columns], df[test_columns],
                                                        test_size=0.5, random_state=98, stratify=df[test_columns])
    return x_train, x_test, y_train, y_test


def knn(x_train, x_test, y_train, y_test, k):
    """
    :param x_train: training set with features
    :param x_test: testing set with features
    :param y_train: training set with labels
    :param y_test: testing set with labels
    :param k: number of neighbors
    :return: accuracy score
    """
    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(x_train)
    test_scaled = scaler.transform(x_test)

    classifier = KNeighborsClassifier(k)
    classifier.fit(train_scaled, y_train.values.ravel())
    y_predict = classifier.predict(test_scaled)

    print(f'KNN: K = {k}, Accuracy = {accuracy_score(y_test, y_predict) * 100:.2f}%')
    return accuracy_score(y_test, y_predict)


def decision_tree(x_train, x_test, y_train, y_test):
    """
    :param x_train: training set with features
    :param x_test: testing set with features
    :param y_train: training set with labels
    :param y_test: testing set with labels
    :return: accuracy score
"""
    # Decision Tree
    dectree = DecisionTreeClassifier(criterion='entropy', random_state=98)
    dectree.fit(x_train, y_train)
    y_predict = dectree.predict(x_test)

    print(f'Accuracy using Decision Tree: {accuracy_score(y_test, y_predict) * 100:.2f}%')


def logistic_regression(x_train, x_test, y_train, y_test):
    """
    :param x_train: training set with features
    :param x_test: testing set with features
    :param y_train: training set with labels
    :param y_test: testing set with labels
    :return: accuracy score
    """
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)

    y_predict = log_reg.predict(x_test)

    print(f'Accuracy using Logistic Regression: {accuracy_score(y_test, y_predict) * 100:.2f}%')


def main():
    df_original = pd.read_csv('customers.csv')
    # make a copy and don't change the original
    df = df_original.copy()
    # print(df.info())

    d_cols = ['ID', 'Customer_ID', 'Month', 'Name', 'Age', 'SSN', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
              'Interest_Rate', 'Type_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
              'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Credit_Mix', 'Credit_Utilization_Ratio',
              'Credit_History_Age', 'Payment_of_Min_Amount', 'Amount_invested_monthly']

    # drop columns not relevant to model
    df = drop_columns(df, d_cols)

    # clean data
    for col in df.columns:
        df = clean_column(df, col)

    # convert numerical columns to floats
    float_cols = ['Annual_Income', 'Outstanding_Debt', 'Monthly_Balance', 'Num_of_Loan',
                  'Total_EMI_per_month', 'Num_Credit_Card']
    df = to_float(df, float_cols)

    # convert categorical data to numerical
    df = cat_to_num(df, 'Payment_Behaviour')

    # drop outliers using IQR
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            df = drop_outlier(df, col)

    # Data should be clean by this point

    # see some plots for cleaned data
    plot_data(df)

    # convert occupation to numerical for training data
    df = cat_to_num(df, 'Occupation')

    train_columns = ['Occupation', 'Annual_Income', 'Num_Credit_Card',
                     'Outstanding_Debt', 'Payment_Behaviour', 'Monthly_Balance', 'Num_of_Loan', 'Total_EMI_per_month']
    test_columns = 'Credit_Score'

    x_train, x_test, y_train, y_test = split_data(df, train_columns, test_columns)

    # test different k values
    k_values = [3, 5, 7, 9, 11, 13, 15]
    accuracies = []
    for i in k_values:
        accuracy = knn(x_train, x_test, y_train, y_test, i)
        accuracies.append(accuracy)

    # plot k values and the corresponding accuracies
    visualize_k(k_values, accuracies)

    logistic_regression(x_train, x_test, y_train, y_test)

    decision_tree(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
