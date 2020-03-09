import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

#Array of columns that are not words
LABEL_COLUMN_IGNORE = ['Full_Text', 'Common_Word_Count', 'Word_Count', 'Date', 'IsSpam', 'Year', 'Month', 'Day']

'''
Formats the content o dataframe.

df: dataFrame with all data.
'''
def format_content(df):
    # Replace 'yes' and 'no' to 1 and 0
    df.loc[df.IsSpam == 'yes', 'IsSpam'] = 1
    df.loc[df.IsSpam == 'no', 'IsSpam'] = 0

    #Generates columns to day month and year
    df['Year'] = pd.DatetimeIndex(df['Date']).year
    df['Month'] = pd.DatetimeIndex(df['Date']).month
    df['Day'] = pd.DatetimeIndex(df['Date']).day

    return df

'''
Get array of words column name.

df: dataFrame with all data.
'''
def get_words_label(df):
    label_words = df.columns.values
    label_words = [x for x in label_words if x not in LABEL_COLUMN_IGNORE]
    return label_words

'''

'''
def print_results_classifier(algorithm_name, val_y, val_predictions):
    print(algorithm_name)
    print("Mean absolute error: {0}".format(mean_absolute_error(val_y, val_predictions)))
    print("Accuracy score: {0}".format(accuracy_score(val_y, val_predictions)))
    print("Confusion matrix: {0}".format(confusion_matrix(val_y, val_predictions)))
    print("Precision score: {0}".format(precision_score(val_y, val_predictions)))
    print("Recal score: {0}".format(recall_score(val_y, val_predictions)))
    print("F1 score: {0}".format(f1_score(val_y, val_predictions)))

'''

'''
def execute_decision_tree_classifier(train_X, val_X, train_y, val_y):
    model = DecisionTreeClassifier(random_state = 0)
    model = model.fit(train_X, train_y)
    val_predictions = model.predict(val_X)
    print_results_classifier('Decision Tree Classifier', val_y, val_predictions)

'''


'''
def execute_random_forest_classifier(train_X, val_X, train_y, val_y):
    model = RandomForestClassifier(random_state = 0)
    model.fit(train_X, train_y)
    val_predictions = model.predict(val_X)
    print_results_classifier('Random Forest Classifier', val_y, val_predictions)

# Main
if __name__ == "__main__":
    # Read csv file =with encoding unicode_escape - this encode was used due to troubles with special characters.
    df = pd.read_csv('../resources/data.csv', encoding='unicode_escape')
    df = format_content(df)
    X = df[get_words_label(df)]
    y = df.IsSpam

    #Transform data from object to int - beacause metrics usage does not accept object as a valid type.
    y = y.astype('int')

    #Give 30% of data for tests
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.3, random_state = 0)

    #Execute both algorithms
    execute_decision_tree_classifier(train_X, val_X, train_y, val_y)
    execute_random_forest_classifier(train_X, val_X, train_y, val_y)