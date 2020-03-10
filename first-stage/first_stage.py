from argparse import ArgumentParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import sys

# Path to output data
PATH_OUTPUT = 'output/'

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

def get_top_words(words_array, ascending=True, n=10):
    '''
    Get the top words most used.

    Parameters:
        words_array(array): array of all words that should be considered.
        ascending(bool): True for ascending and False for descending. Default: True.
        n(int): number of top words that should be returned. Default: 10.
    Returns:
        Array: Top N values sorted in ascending/descending order.
    '''
    if (n <= 0):
        raise Exception("n can't be less or equal than 0.")

    return words_array.sort_values(ascending=ascending)[0:n]


def plot_graph_bar(label_words, total_words, file_name='graph_bar'):
    '''
    Plots the data into a bar graph
    
    Parameters:
        label_words (Array): label of all words in an array
        total_words (Array): sum of all words in an array
        file_name(str): name of file to save on disk (Optional).
    
    Returns:
        None
    '''
    index = np.arange(len(label_words))
    plt.figure(figsize=(80,30))
    bars = plt.bar(index, total_words, align='center', width=0.5)
    plt.xlabel('Words', fontsize=8)
    plt.ylabel('Occurence Number', fontsize=8)
    plt.xticks(index, label_words, fontsize=8, rotation=60)
    plt.title('Word count')

    for bar in bars:
        val = bar.get_height()
        plt.text(bar.get_x(), val + .002, val)

    # Uncomment this lines to save locally
    # file_name = '{0}/{1}.png'.format(PATH_OUTPUT, file_name)
    # plt.savefig(file_name)
    plt.show()

def generate_wordcloud(label_words, total_words):
    '''
    Generates a wordcloud of all words

    Parameters:
        label_words(Array): label of all words in an array
        total_words(Array): sum of all words in an array

    Returns: None
    '''

    #Generates a dict from label and total words. The dict must be: ('word', Number o occurences)
    word_dict = {label_words[i] : total_words[i] for i in range(0, len(label_words))}

    wordcloud = WordCloud(background_color='white', 
    width=1600, 
    height=800).generate_from_frequencies(word_dict)

    # Uncomment this two lines to save on disk
    # filename = '{0}/{1}.png'.format(PATH_OUTPUT, 'wordcloud')
    # wordcloud.to_file(filename)
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()


def get_messages_sum_of_month(df, month=1, is_spam=False):
    '''
    Get the quantity of messages for a specific month

    Parameters:
        df(DataFrame): DataFrame with all data
        month(int): Month selected. Default = 1
        is_spam(bool): Filter spam or common. Default = False (Common)
    Returns:
        DataFrame: filtered information.

    '''
    data = df[(df.IsSpam == is_spam) & (df.Month == month)]
    return data

def plot_graph_messages_of_month(df, label_words, month=1, is_spam=False):
    '''
    Generates the graph bar of messages of a specific month, with spam or common messages.

    Parameters:
        df(DataFrame): DataFrame with all data
        label_words(Array): all labels of words that should be considered
        month(int): Month selected. Default = 1
        is_spam(bool): Filter spam or common. Default = False (Common)
    Returns:
        None
    '''
    data = get_messages_sum_of_month(df, month, is_spam)

    if (len(data) > 0):
        total_words = data[label_words].sum()
        file_name = 'month_{0}_report_{1}_messages'.format(month, 'spam' if is_spam else 'common')
        plot_graph_bar(label_words, total_words, file_name)
    else:
        print("Ignoring month:{0} with is_spam={1} beacause it is empty.".format(month, is_spam))

def generate_graph_for_all_months(df, label_words, is_spam=False):
    '''
    Generate graph for all months.

    Parameters:
        df(DataFrame): DataFrame with all data
        label_words(Array): all labels of words that should be considered
        month(int): Month selected. Default = 1
        is_spam(bool): Filter spam or common. Default = False (Common)
    Returns:
        None
    '''
    for month in range(1, 13):
        plot_graph_messages_of_month(df, label_words, month, is_spam)

def generate_graph_for_all_spam_common_months(df, label_words):
    '''
    Generate graph for all month - spam and common messages

    Parameters:
        df(DataFrame): DataFrame with all data
        label_words(Array): all labels of words that should be considered
    Returns:
        None
    '''
    data = df.groupby(['Month', 'IsSpam'])['Month', 'IsSpam']
    data.size().unstack().plot.bar()
    plt.show()

def print_statistical_data_month(df, label_words, month=1, n_digits=2):
    '''
    Print statical data (Max, min, mean, median, std and variance) based on specific month.

    Parameters:
        df(DataFrame): DataFrame with all data
        label_words(Array): all labels of words that should be considered
        month(int): Month selected. Default = 1
        n_digits(int): Number of digits that should be rounded. Default = 2.

    Returns:
        None
    '''
    data = df[df.Month == month]
    if (len(data) > 0):
        print("Month: {0}".format(month))
        print("Max: {0}".format(data.Word_Count.max()))
        print("Min: {0}".format(data.Word_Count.min()))
        print("Mean: {0}".format(round(data.Word_Count.mean(), n_digits)))
        print("Median: {0}".format(round(data.Word_Count.median(), n_digits)))
        print("STD: {0}".format(round(data.Word_Count.std(), n_digits)))
        print("Variance: {0}".format(round(data.Word_Count.var(), n_digits)))
        print("\n")
    else:
        print("Ignoring month:{0} beacause it is empty.".format(month))

def print_statistical_data_all_month(df, label_words):
    '''
    Print statical data (Max, min, mean, median, std and variance) of all month's

    Parameters:
        df(DataFrame): DataFrame with all data
        label_words(Array): all labels of words that should be considered
    
    Returns:
        None
    '''
    for month in range(1, 13):
        print_statistical_data_month(df, label_words, month)

def get_day_of_month_sequence_common_message(df, label_words, month=1):
    '''
    Get the sequence which has the most sequence of common messages of a specific month.

    Parameters:
        df(DataFrame): DataFrame with all data
        label_words(Array): all labels of words that should be considered
        month(int): Month selected. Default = 1
    Returns:
        None

    '''
    data = df[df.Month == month]

    if (len(data) <= 0):
        print("Ignoring month:{0} beacause it is empty.".format(month))
        return

    min_day = data.Day.min()
    max_day = data.Day.max()

    max_value = 0
    max_value_day = 0

    for day in range(min_day, max_day + 1):
        data_day = data[data.Day == day]

        sequence = 0
        for _, row in data_day.iterrows():
            if (row.IsSpam == False):
                sequence += 1
            else:
                if (sequence > max_value):
                    max_value = sequence
                    max_value_day = day
                    sequence = 0
                else:
                    sequence = 0
        
    print("Most sequence without spam messages is: {0} of month {1} and day {2}.".format(max_value, month, max_value_day))

def get_day_of_month_sequence_common_message_all(df, label_words):
    '''
    Get the sequence which has the most sequence of common messages of all months.

    Parameters:
        df(DataFrame): DataFrame with all data
        label_words(Array): all labels of words that should be considered
    Returns:
        None
    '''
    for month in range(1, 13):
        get_day_of_month_sequence_common_message(df, label_words, month)


def parse_args():
    '''
        Parse Arguments from CLI.
    '''
    ap = ArgumentParser()

    ap.add_argument(
        '--o',
        '-option',
        required=True,
        type=int,
        help='Should be an option based on the question number. \n 1) Show graph most frequency words \n 2) Show Wordcloud most frequency words 3) Show quantity message common and spam per month 4) Show statistical data (max, min, mean, median, std, variance and total words for each month) 5) Show day with more sequence of commom messages'
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
    label_words = get_words_label(df)
    total_words = df[label_words].sum()

    # Show graph most frequency words
    if args.o == 1:
        plot_graph_bar(label_words, total_words)

    # Show Wordcloud most frequency words
    elif args.o == 2:
        generate_wordcloud(label_words, total_words)

    # Show quantity message common and spam per month
    elif args.o == 3:
        generate_graph_for_all_spam_common_months(df, label_words)

    # Show statistical data (max, min, mean, median, std, variance and total words for each month)
    elif args.o == 4:
        print_statistical_data_all_month(df, label_words)

    # Show day with more sequence of commom messages
    elif args.o == 5:
        get_day_of_month_sequence_common_message_all(df, label_words)

    # Uncomment this lines if you want the secret 6th option with top N most used words. This was used only for tests.
    # Show top 10 words
    # elif args.o == 6:
    #     print(get_top_words(total_words, ascending=False))
    
    else:
        raise Exception("None of accepted options were inserted. Values of range [1-5] only accepted.")

#
# The program starts here \/
#
if __name__ == "__main__":
    args = parse_args()
    main(args)
    sys.exit(0)