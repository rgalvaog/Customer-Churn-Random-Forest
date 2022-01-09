# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

### Summary
This project conducts a machine learning analysis of customer churn. In addition to this README.md file, I include the main script 'churn_library.py' and the log test 'churn_script_logging_and_testing.py'. In this project, I conduct EDA to show both categorical and continuous features of the dataset, then train the model with both a logistic regression and random forest classifiers. This project was the first assignment in Udacity's 'Machine Learning DevOps Engineer' nanodegree. 

### Output
Four visualizations in the eda subfolder within the images folder:

    (1) churn_hist.png
    (2) customerAge_hist.png
    (3) heatmap_correlations.png
    (4) maritalStatus_bar.png

Four visualizations in the results subfolder within the images folder:

    (1) classification_report_lr.png
    (2) classification_report_rf
    (3) feature_importance.png
    (4) roc_curves.png

Two model objects in the models folder:

    (1) logistic_model.pkl
    (2) rfc_model.pkl

## Running Files

Below are instructions to run the files:

1. If running on local machine: 
```python3 churn_library.py```
   
2. If running on Udacity's patform:  
```ipython churn_library.py```  
   
4. To run unit tests on all major functions of churn_library.py 
```python3 churn_script_logging_and_tests.py```
   
Output: Images generated with the script will be available in the images folder, the models generated will be available in the models folder, and the log will be available in the, you guessed it, logs folder.
