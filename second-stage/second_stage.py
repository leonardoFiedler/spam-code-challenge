from argparse import ArgumentParser
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import sys

#Array of columns that are not words
LABEL_COLUMN_IGNORE = ['Full_Text', 'Common_Word_Count', 'Word_Count', 'Date', 'IsSpam', 'Year', 'Month', 'Day']

def format_content(df):
    '''
    Formats the content o dataframe.

    Parameters:
        df (DataFrame): dataFrame with all data.

    Returns:
        DataFrame: dataFrame with all formated data.
    '''

    # Replace 'yes' and 'no' to 1 and 0
    df.loc[df.IsSpam == 'yes', 'IsSpam'] = 1
    df.loc[df.IsSpam == 'no', 'IsSpam'] = 0

    #Generates columns to day month and year
    df['Year'] = pd.DatetimeIndex(df['Date']).year
    df['Month'] = pd.DatetimeIndex(df['Date']).month
    df['Day'] = pd.DatetimeIndex(df['Date']).day

    return df

def get_words_label(df):
    '''
    Get array of words column name.

    Parameters:
        df(DataFrame): dataFrame with all data.
    
    Returns:
        Array: Array of all words column name
    '''

    label_words = df.columns.values
    label_words = [x for x in label_words if x not in LABEL_COLUMN_IGNORE]
    return label_words

def print_results_classifier(algorithm_name, val_y, val_predictions, n_digits=2):
    '''
    Print the result of a classifier. It will print: Mean absolute error, Accuracy score, Confusion matrix, 
    Precision score, Recal score, F1 score

    Parameters:
        algorithm_name(str): Name of algorithm - just to print
        val_y(Array): labels of y (right labels)
        val_predictions(Array): labels of y (predicted labels)
        n_digits(int): number of digits to round the function. Default = 2
    
    Returns:
        None
    '''
    print(algorithm_name)
    print("Mean absolute error: {0}".format(round(mean_absolute_error(val_y, val_predictions), n_digits)))
    print("Accuracy score: {0}".format(round(accuracy_score(val_y, val_predictions), n_digits)))
    print("Confusion matrix: {0}".format(confusion_matrix(val_y, val_predictions)))
    print("Precision score: {0}".format(round(precision_score(val_y, val_predictions), n_digits)))
    print("Recal score: {0}".format(round(recall_score(val_y, val_predictions), n_digits)))
    print("F1 score: {0}".format(round(f1_score(val_y, val_predictions), n_digits)))
    print("\n")

def execute_decision_tree_classifier(train_X, val_X, train_y, val_y):
    '''
    Execute the decision tree classifier, fit and predict.

    Parameters:
        train_X(matrix): X data for training
        val_X(matrix): X data for validation
        train_y(array): y labels for training
        val_y(array): y labels for validation

    Results:
        None
    '''
    model = DecisionTreeClassifier(random_state = 0)
    model = model.fit(train_X, train_y)
    val_predictions = model.predict(val_X)
    print_results_classifier('Decision Tree Classifier', val_y, val_predictions)

def execute_random_forest_classifier(train_X, val_X, train_y, val_y):
    '''
    Execute the random forest classifier, fit and predict.

    Parameters:
        train_X(matrix): X data for training
        val_X(matrix): X data for validation
        train_y(array): y labels for training
        val_y(array): y labels for validation

    Results:
        None
    '''
    model = RandomForestClassifier(random_state = 0)
    model.fit(train_X, train_y)
    val_predictions = model.predict(val_X)
    print_results_classifier('Random Forest Classifier', val_y, val_predictions)

def parse_args():
    '''
        Parse Arguments from CLI.
    '''
    ap = ArgumentParser()

    ap.add_argument(
        '--o',
        '-option',
        required=False,
        type=int,
        help='Should be an option based on the algorithm number. \n 1) Random Forest \n 2) Decision Tree.'
    )

    ap.add_argument(
        '--a',
        '-all',
        required=False,
        action='store_true',
        help='Execute both algorithms.'
    )

    return ap.parse_args()

def main(args):
    '''
    Main function that should be invoked.

    Parameters:
        args(ArgumentParser): Arguments passed from CLI.
    
    Returns:
        None

    Raise:
        Exception: if none of the options match.
    '''

    # Read csv file =with encoding unicode_escape - this encode was used due to troubles with special characters.
    df = pd.read_csv('../resources/data.csv', encoding='unicode_escape')
    df = format_content(df)
    X = df[get_words_label(df)]
    y = df.IsSpam

    #Transform data from object to int - beacause metrics usage does not accept object as a valid type.
    y = y.astype('int')

    #Split data and give 30% of data for tests
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.3, random_state = 0)
    
    if args.o is not None:
        if args.o == 1:
            execute_random_forest_classifier(train_X, val_X, train_y, val_y)
            return
        elif args.o == 2:
            execute_decision_tree_classifier(train_X, val_X, train_y, val_y)
            return
        else:
            raise Exception("None of accepted options were inserted. Values of range [1-2] only accepted.")

    # Execute both
    if args.a is not None:
        execute_random_forest_classifier(train_X, val_X, train_y, val_y)
        execute_decision_tree_classifier(train_X, val_X, train_y, val_y)
        return

    raise Exception("None of parameters passed. Use python second_stage -h for help of accepted parameters.")
#
# The program starts here \/
#
if __name__ == "__main__":
    args = parse_args()
    main(args)
    sys.exit(0)