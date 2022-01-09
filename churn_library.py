'''
Predict Customer Churn with Clean Code
Rafael Guerra
Jan 7, 2021

This Python script conducts a machine learning analysis of customer churn.
The following parts are included in this script:

-EDA
-Feature Engineering
-Model Training
-Prediction
-Model Evaluation

'''

# Import Libraries

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


# Functions

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            dataframe: pandas dataframe
    '''
    try:
        dataset = pd.read_csv(pth)
        return dataset
    except FileNotFoundError:
        return 'Sorry! The filepath provided does not exist.'


def perform_eda(dataframe):
    '''
    perform eda on dataframe and save figures to images folder
    input:
            dataframe: pandas dataframe from import_data

    output:
            None
    '''
    # Create Churn feature
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    # Set Visualization Parameters

    viz_x_dim = 20
    viz_y_dim = 10

    # Generate Histogram for Churn
    plt.figure(figsize=(viz_x_dim, viz_y_dim))
    dataframe['Churn'].hist()
    plt.savefig('./images/eda/churn_hist.png')

    # Generate Histogram for Customer Age
    plt.figure(figsize=(viz_x_dim, viz_y_dim))
    dataframe['Customer_Age'].hist()
    plt.savefig('./images/eda/customerAge_hist.png')
    
    # Generate Bar Chart for Marital Status
    plt.figure(figsize=(viz_x_dim, viz_y_dim))
    dataframe.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/maritalStatus_bar.png')

    # Generate Heatmap for Correlation Matrix
    plt.figure(figsize=(viz_x_dim, viz_y_dim))
    sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/eda/heatmap_correlations.png')


def encoder_helper(dataframe, category_lst, new_col_suffix):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name

    output:
            dataframe: pandas dataframe with new columns
    '''
    try:
        # Create Churn Variable
        dataframe['Churn'] = dataframe['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

        # Create new columns with mean of each categorical variable
        for category in category_lst:
            category_lst = []
            category_mean_by_group = dataframe.groupby(category).mean()['Churn']
            for entry in dataframe[category]:
                category_lst.append(category_mean_by_group.loc[entry])
            dataframe[str(category) + new_col_suffix] = category_lst

        # Return new column dataframe with new features
        return dataframe

    except TypeError:
        return 'Function args expect dataframe, list, and string. Check your types.'


def perform_feature_engineering(dataframe, response):
    '''
    input:
              dataframe: pandas dataframe
              response: string of response name

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    try:
        # Create X dataframe and y outcome variable
        x_dataset = pd.DataFrame()
        y_variable = dataframe[response]

        # Keep only relevant columns
        columns_to_keep = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                           'Total_Relationship_Count', 'Months_Inactive_12_mon',
                           'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                           'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                           'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                           'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                           'Income_Category_Churn', 'Card_Category_Churn']
        x_dataset[columns_to_keep] = dataframe[columns_to_keep]

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(x_dataset, y_variable, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    except TypeError:
        return 'Function args expect dataframe and str. Check your types.'


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # Random Forest Classification Report
    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_rf)), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_rf)), {
        'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/classification_report_rf.png')

    # Logistic Regression Classification Report
    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
        'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/classification_report_lr.png')


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # Set up two models: Random Forest, and Log Reg
    rfc = RandomForestClassifier()
    lrc = LogisticRegression(solver='liblinear', max_iter=1000)

    # Parameters for Grid Search
    # Note: I have shrunk the parameter space so the script could run faster
    param_grid = {
        'n_estimators': [10, 100],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 500],
        'criterion': ['gini']
    }
    
    # Train Random Forest and generate predictions
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    
    # Train Log Regression and generate predictions
    lrc.fit(X_train, y_train)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    
    # Create and save models in models folder
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    # Generate ROC Curves
    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)
    plt.figure(figsize=(15, 8))
    plot_roc_curve(rfc_model, X_test, y_test, ax=plt.gca(), alpha=0.8)
    lrc_plot.plot(ax=plt.gca(), alpha=0.8)
    plt.savefig('./images/results/roc_curves.png')
    
    # Generate classification report image
    classification_report_image(y_train, y_test, y_train_preds_lr,
                                y_train_preds_rf, y_test_preds_lr,
                                y_test_preds_rf)
    
    # Generate feature importance plot
    feature_importance_plot(rfc_model, X_test, './images/results/feature_importance.png')


if __name__ == "__main__":
    bank_data = import_data(r"./data/bank_data.csv")
    perform_eda(bank_data)
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    NEW_COL_SUFFIX = '_Churn'
    OUTCOME_VARIABLE = 'Churn'
    encoded_data = encoder_helper(bank_data, cat_columns, NEW_COL_SUFFIX)
    X_training_set, X_test_set, y_training_set, y_test_set = perform_feature_engineering(encoded_data, OUTCOME_VARIABLE)
    train_models(X_training_set, X_test_set, y_training_set, y_test_set)
