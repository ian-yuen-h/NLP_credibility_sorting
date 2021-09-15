# import libraries, modules & packages
import random
import datetime
import pandas as pd

# libraries for machine learning and model evaluation
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# libraries for dealing with imbalanced data
import numpy as np
from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

# libraries for visualization
import seaborn as sb
import matplotlib.pyplot as plt


'''fit a defined model'''
def train_model(model):
    model.fit(X_train.fillna(0), y_train.values.ravel())


'''get the set of labels predicted by the model'''
def get_predicted_labels(model, X_test):
    labels = model.predict(X_test)
    return labels  


'''calculate and print the accuracy score of a model'''
def print_model_accuracy(y_test, y_pred):  # y_test is already defined, y_pred is specific to each model
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {accuracy}")


'''print feature importance scores'''
def print_importance_scores(importance_scores):
    for i,v in enumerate(importance_scores):
        print('Feature: %0d, Score: %.5f' % (i,v))


'''calculate & graph feature importance scores using the classifier's feature_importances_ component'''
def display_model_feature_importance(model, features):
    print("Feature Importance from Model")
    print('-'*29)
    # use one of the ff 3 lines
    importance_scores = pd.Series(model.feature_importances_, index=features) #.sort_values(ascending=False)
    # importance_scores = model.best_estimator_._final_estimator.feature_importances_
    # importance_scores = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
    print_importance_scores(importance_scores)  # print features & scores
    # create bar graph and display features & their scores using seaborn
    sb.barplot(x=importance_scores, y=features)  # alternatively, y=importance_scores.index
    # add labels
    plt.title("Feature Importance Graph")
    plt.xlabel('Importance Scores')
    plt.ylabel('Features')
    plt.show()


'''calculate and graph feature importances using permutation_importance'''
def display_permutation_feature_importance(model, X):
    print("Permutation Feature Importance")
    print('-'*30)
    # calculate permutation importance
    perm_imp = permutation_importance(model, X.fillna(0), y, scoring='accuracy')
    importance_scores = perm_imp.importances_mean
    print_importance_scores(importance_scores)  # print features & scores
    # create bar graph and display features & their scores using seaborn
    sb.barplot(x=importance_scores, y=list(X.columns))
    # add labels
    plt.title('Permutation Importance Graph')
    plt.xlabel('Importance Scores')
    plt.ylabel('Features')
    plt.show()


'''get the feature importance based on Mean Decrease in Impurity'''
def display_mdi_feature_importance(model, features):
    print("Feature Importance based on Mean Decrease in Impurity(MDI)")
    print('-'*59)
    importances = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    importance_scores = pd.Series(importances, index=features)
    model_importances = pd.Series(importances, index=features)
    # plot the features and their scores on a bar graph with matplotlib
    fig, ax = plt.subplots()
    model_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()


'''get classification report of model'''
def display_classification_report(y_pred):  # get the labels predicted by a model, pass to the function & print the classification report
    print("Classification Report:")
    print('-'*22)
    y_true = [random.randint(0,2) for _ in range(len(y_pred))]  # randomly generated true values
    target_names = ['medium', 'high']  # modify this list if low credibility articles are included
    print(classification_report(y_true, y_pred, target_names=target_names, labels=[1,2]))


'''evaluate model using confusion matrix output'''
def create_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, columns=['Predicted Negative', 'Predicted Positive'], index=['Actual Negative', 'Actual Positive'])
    return cm_df  # return value needed for classification report


'''report metrics using values from the confusion matrix'''
def report_classification_metrics(cm):
    print("\nClassification Metrics:")
    print('-'*23)
    TP = cm['Predicted Positive'][1]  # True Positives
    TN = cm['Predicted Negative'][0]  # True Negatives
    FP = cm['Predicted Positive'][0]  # False Positives
    FN = cm['Predicted Negative'][1]  # False Negatives
    # calculate True & False  Positive Rates
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    # calculate the six metrics
    total = sum([TP, TN, FP, FN])
    accuracy = (float(TP+TN)/float(total))
    precision = (TP/float(TP+FP))
    sensitivity = (TP/float(FN+TP))
    specificity = (TN/float(TN+FP))
    f1_score = 2 * ((precision * sensitivity)/(precision + sensitivity))
    mis_classification = 1 - accuracy
    # print results
    print(f'True Positives: {TP}')
    print(f'True Negatives: {TN}')
    print(f'False Positives: {FP}')
    print(f'False Negatives: {FN}')
    print()
    print(f'True Positive Rate: {TPR}')
    print(f'False Positive Rate: {FPR}')
    print()    
    print("\nThe 6 metrics:")
    print('-'*15)
    print(f'F1 Score: {round(f1_score, 2)}')
    print(f'Accuracy: {round(accuracy, 2)}')
    print(f'Precision: {round(precision, 2)}')
    print(f'Specificity: {round(specificity, 2)}')
    print(f'Sensitivity: {round(sensitivity, 2)}')
    print(f'Mis-classification: {round(mis_classification, 2)}')


'''calculate & print the roc auc score'''
def print_roc_auc_score(model, X, y):
    score = roc_auc_score(y, model.predict_proba(X.fillna(0))[:, 1])
    print(f'ROC AUC Score: {score}')


'''calculate & print the mean roc auc score using k-fold cross-validation'''
def print_mean_roc_auc_score(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X.fillna(0), y, scoring='roc_auc', cv=cv, n_jobs=-1)
    mean_score = mean(scores)
    print(f'Mean ROC AUC Score: {mean_score}')


'''take a model and plot its ROC curve'''
def plot_roc_curve(model, X_test, y_test):
    metrics.plot_roc_curve(model, X_test, y_test)
    plt.title('ROC Curve')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


'''analyze a model. the only function to use, after you initialize the model'''
def analyze_model(model, X, X_test, y_test, features):
    train_model(model)
    y_pred = get_predicted_labels(model, X_test)
    print_model_accuracy(y_test, y_pred)
    print()
    # comment out feature importance code for Easy Ensemble & BalancedBaggingClassifier
    display_model_feature_importance(model, features)
    display_permutation_feature_importance(model, X)
    display_mdi_feature_importance(model, features)
    print()
    display_classification_report(y_pred)
    cm = create_confusion_matrix(y_test, y_pred)
    print()
    print("Confusion Matrix:")
    display(cm)
    report_classification_metrics(cm)
    print()
    print_roc_auc_score(model, X, y)
    print()
    print_mean_roc_auc_score(model, X, y)
    print()
    plot_roc_curve(model, X_test, y_test)



# create and analyze models in the main function below
def main():
    # load & prepare dataset
    df = pd.read_csv('dataset.csv')
    df = df.fillna(0)
    
    # map cred_score column values to ints
    df['cred_score'] = df['cred_score'].map({'medium':1, 'high':2})
    
    # load and remove unwanted columns from features list
    f = list(df.columns.values)
    f.remove('id')
    f.remove('publisher')
    f.remove('entity_dict')
    f.remove('cred_score')
    f.remove('publisher_val')
    print(f)
    
    # define features and labels
    features = f
    labels = ['cred_score']
    X = df[features]
    y = df[labels]
    
    # split dataset to training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)  # 10% testing and 90% training

    # define and analyze standard Random Forest
    model = RandomForestClassifier(n_estimators=100)
    analyze_model(model, X, X_test, y_test, features)
    
    
    '''Dealing with imbalanced data'''
"""   
    # Standard Bagging
    bc_model = BaggingClassifier(n_estimators=100)
    analyze_model(bc_model, X, X_test, y_test, features)
    
    # Random Forest with Class Weighting
    rf_cw_model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    analyze_model(rf_cw_model, X, X_test, y_test, features)

    # Random Forest with Bootstrap Class Weighting
    rf_bcw_model = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample')
    analyze_model(rf_bcw_model, X, X_test, y_test, features)

    # Random Forest with Random Undersampling
    brf_model = BalancedRandomForestClassifier(n_estimators=100)
    analyze_model(brf_model, X, X_test, y_test, features)

    ### comment the 3 feature importance lines in the **analyze_model** function before running the following cells
    
    # Bagging with Random Undersampling
    bb_model = BalancedBaggingClassifier(n_estimators=100)
    analyze_model(bb_model, X, X_test, y_test, features)
    
    # Easy Ensemble
    ee_model = EasyEnsembleClassifier(n_estimators=100)
    analyze_model(ee_model, X, X_test, y_test, features)
""" 

if __name__ == '__main__':
    main()
