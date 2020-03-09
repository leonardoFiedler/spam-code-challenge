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
Get the top words most used.

words_array: array of all words that should be considered.
ascending: True for ascending and False for descending. Default: True.
n: number of top words that should be returned. Default: 10.
'''
def get_top_words(words_array, ascending=True, n=10):
    if (n <= 0):
        raise Exception("n can't be less or equal than 0.")

    return words_array.sort_values(ascending=ascending)[0:n]

'''
Plot in ascending/descending order

total_words: sum of all words in an array
'''
# def plot_graph_bar_ordered(total_words):
#     plt.figure(figsize=(50,10))
#     plt.xlabel('Words', fontsize=8)
#     plt.xticks(rotation=60)
#     plt.ylabel('Occurence Number', fontsize=8)
#     total_words.sort_values().plot.bar()
#     plt.title('Word Count Ordered')
#     plt.savefig(PATH_OUTPUT + 'graph_bar_ordered.png')

'''
Plots the data into a bar graph
label_words = label of all words in an array
total_words = sum of all words in an array
'''
def plot_graph_bar(label_words, total_words, file_name='graph_bar'):
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

    
    # file_name = '{0}/{1}.png'.format(PATH_OUTPUT, file_name)
    # plt.savefig(file_name)
    plt.show()

'''
Generates a wordcloud of all words

label_words = label of all words in an array
total_words = sum of all words in an array
'''
def generate_wordcloud(label_words, total_words):
    #Generates a dict from label and total words. The dict must be: ('word', Number o occurences)
    word_dict = {label_words[i] : total_words[i] for i in range(0, len(label_words))}

    wordcloud = WordCloud(background_color='white', 
    width=1600, 
    height=800).generate_from_frequencies(word_dict)
    # filename = '{0}/{1}.png'.format(PATH_OUTPUT, 'wordcloud')
    # wordcloud.to_file(filename)
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()

'''
Get the quantity of messages for a specific month

'''
def get_messages_sum_of_month(df, month=1, is_spam=False):
    data = df[(df.IsSpam == is_spam) & (df.Month == month)]
    return data

'''
Generates the graph bar of messages of a specific month, with spam or common messages.


'''
def plot_graph_messages_of_month(df, label_words, month=1, is_spam=False):
    data = get_messages_sum_of_month(df, month, is_spam)

    if (len(data) > 0):
        total_words = data[label_words].sum()
        file_name = 'month_{0}_report_{1}_messages'.format(month, 'spam' if is_spam else 'common')
        plot_graph_bar(label_words, total_words, file_name)
    else:
        print("Ignoring month:{0} with is_spam={1} beacause it is empty.".format(month, is_spam))

'''
Generate graph for all months.

'''
def generate_graph_for_all_months(df, label_words, is_spam=False):
    for month in range(1, 13):
        plot_graph_messages_of_month(df, label_words, month, is_spam)

'''
Generate graph for all month - spam and common messages

'''
def generate_graph_for_all_spam_common_months(df, label_words):
    data = df.groupby(['Month', 'IsSpam'])['Month', 'IsSpam']
    data.size().unstack().plot.bar()
    plt.show()

'''
Print statical data (Max, min, mean, median, std and variance) based on specific month.
'''
def print_statistical_data_month(df, label_words, month=1):
    data = df[df.Month == month]
    if (len(data) > 0):
        print("Month: {0}".format(month))
        print("Max: {0}".format(data.Word_Count.max()))
        print("Min: {0}".format(data.Word_Count.min()))
        print("Mean: {0}".format(data.Word_Count.mean()))
        print("Median: {0}".format(data.Word_Count.median()))
        print("STD: {0}".format(data.Word_Count.std()))
        print("Variance: {0}".format(data.Word_Count.var()))
        print("\n")
    else:
        print("Ignoring month:{0} beacause it is empty.".format(month))

'''
Print statical data (Max, min, mean, median, std and variance) of all month's

'''
def print_statistical_data_all_month(df, label_words):
    for month in range(1, 13):
        print_statistical_data_month(df, label_words, month)


'''
Get the sequence which has the most sequence of common messages of a specific month.

'''
def get_day_of_month_sequence_common_message(df, label_words, month=1):
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

'''
Get the sequence which has the most sequence of common messages of all months.

'''
def get_day_of_month_sequence_common_message_all(df, label_words):
    for month in range(1, 13):
        get_day_of_month_sequence_common_message(df, label_words, month)

def parse_args():
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
    
    else:
        raise Exception("None of accepted options were inserted. Values of range [1-5] only accepted.")
    
# Main
if __name__ == "__main__":
    # Question 01
    # total_words = df[label_words].sum()
    # top_10_words = get_top_words(total_words, ascending=False)
    # print(top_10_words)

    # Generate bar graph
    # plot_graph_bar(label_words, total_words)

    # Generate wordcloud
    # generate_wordcloud(label_words, total_words)

    #
    # Question 02
    #

    # print(df[['Date', 'IsSpam']])
    # plot_graph_messages_of_month(df, label_words, is_spam=False, month=2)
    # generate_graph_for_all_spam_common_months(df, label_words)

    #
    # Question 03
    #
    # print_statistical_data_all_month(df, label_words)

    # Question 04
    # get_day_of_month_sequence_common_message(df, label_words, month=3)
    # get_day_of_month_sequence_common_message_all(df, label_words)
    args = parse_args()
    main(args)
    sys.exit(0)

# Mean of each word
# print("Mean")
# print(df[label_words].mean())

# Standard deviation of each word
# print("Standard Deviation")
# print(df[label_words].std())