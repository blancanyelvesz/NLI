# import libraries
import os
import pandas as pd # type: ignore
import nltk # type: ignore
from nltk import word_tokenize, pos_tag # type: ignore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore 
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')

data_directory = 'data'

def read_file(filename):
    filepath = os.path.join(data_directory, filename)
    with open(filepath, 'r', encoding='utf-8') as file:
        filetext = file.read()
    return filetext


def get_pos_tags(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    pos_tag_str = " ".join([tag for _, tag in pos_tags])
    return pos_tag_str


analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']


def choose_method(choice):
    if choice == 'SV':
        method = LinearSVC(multi_class = 'ovr', max_iter = 10000)
    elif choice == 'RF':
        method = RandomForestClassifier(class_weight = 'balanced')
    elif choice == 'KN':
        method = KNeighborsClassifier(n_neighbors = 10)
    else:
        raise TypeError("The only methods available are 'KN', 'RF' and 'SV'!")
    return method


results_directory = 'results'
os.makedirs(results_directory, exist_ok=True)

def save_report(y_test, y_pred, filename):
    reportpath = os.path.join(results_directory, f'report_{filename}.csv')
    report = classification_report(y_test, y_pred, output_dict = True, zero_division = 0)
    results = pd.DataFrame(report).transpose()
    results.to_csv(reportpath) 


def show_and_save_matrix(y_test, y_pred, classifier, filename):
    confmatrix = confusion_matrix(y_test, y_pred, labels = classifier.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix = confmatrix, display_labels = classifier.classes_)
    fig, ax = plt.subplots(figsize=(10, 7))
    display.plot(ax=ax)
    ax.set_title(filename)
    filepath = os.path.join(results_directory, f'confusion_matrix_{filename}.png')
    fig.savefig(filepath)
    plt.show()
    
    