'''
Predict Customer Churn with Clean Code
Rafael Guerra
Jan 7, 2021

This Python script conducts unit tests and logging.

'''
# Libraries
import os
import logging
import churn_library as cls

# Create .log
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# Functions

def test_import(dataset):
    '''
    test data import
    '''
    try:
        imported_df = cls.import_data(dataset)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert imported_df.shape[0] > 0
        assert imported_df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return imported_df


def test_eda(perform_eda, dataframe):
    '''
    test perform eda function
    '''
    # Log success of function
    try:
        perform_eda(dataframe)
        logging.info("Testing perform_eda: SUCCESS")
    except FileNotFoundError as err:
        logging.error("FileNotFoundError: The file entered was not found or does not exist.")
        raise err

    # Log that output files have been properly created
    try:
        assert os.path.exists('./images/eda/churn_hist.png')
        assert os.path.exists('./images/eda/customerAge_hist.png')
        assert os.path.exists('./images/eda/heatmap_correlations.png')
        assert os.path.exists('./images/eda/maritalStatus_bar.png')
    except AssertionError as err:
        logging.error(f"AssertionError: One or more charts have not been created:{err}")
        raise err


def test_encoder_helper(encoder_helper, data, cat_columns):
    '''
    test encoder helper
    '''
    try:
        encoder_helper(data, cat_columns, '_Churn')
        logging.info("Testing encoding: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoding: FAILURE")
        raise err


def test_perform_feature_engineering(perform_feature_engineering, data):
    '''
    test perform_feature_engineering
    '''
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(data, 'Churn')
        logging.info("Testing feature engineering: SUCCESS")

    except AssertionError as err:
        logging.error("Testing feature engineering: FAILURE")
        raise err

    return X_train, X_test, y_train, y_test


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    test train_models
    '''
    try:
        train_models(X_train, X_test, y_train, y_test)
        logging.info("Testing model training: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing model training: FAILURE")
        raise err

    try:
        # Assert models
        assert os.path.exists('./models/logistic_model.pkl')
        assert os.path.exists('./models/rfc_model.pkl')

        # Assert Visualizations
        assert os.path.exists('./images/results/classification_report_lr.png')
        assert os.path.exists('./images/results/classification_report_rf.png')
        assert os.path.exists('./images/results/feature_importance.png')
        assert os.path.exists('./images/results/roc_curves.png')
    except AssertionError as err:
        logging.error(f"AssertionError: One or more charts have not been created:{err}")
        raise err


if __name__ == "__main__":
    BANK_DATASET = "./data/bank_data.csv"
    imported_dataset = test_import(BANK_DATASET)
    test_eda(cls.perform_eda, imported_dataset)
    category_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    test_encoder_helper(cls.encoder_helper, imported_dataset, category_columns)
    xtrain, xtest, ytrain, ytest = test_perform_feature_engineering(cls.perform_feature_engineering,
                                                                    imported_dataset)
    test_train_models(cls.train_models, xtrain, xtest, ytrain, ytest)
