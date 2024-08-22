""" Native Language Identification Task Using Different Algorithms

Usage:
  nli_project.py [options]
  nli_project.py (-h | --help)
  nli_project.py --version

Options:
  -a --algorithm (KN|RF|SV)     Algorithm to use in the task.  
                                Can be k-Neighbors (KN), Random Forest (RF), or Support Vector (SV).
  -l --levels       Separate the task according to the language level of the authors of the texts. [default: False]
  -s --save         Save results report as .csv file and confusion matrix as .png file. [default: False]
  -e --emo          Include sentiment analysis in the feature extraction. [default: True]
  -h --help         Show this screen.
  --version         Show version.
"""

from docopt import docopt
import pandas as pd # type: ignore
import numpy as np
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.sparse import vstack, hstack
from scipy.sparse import csr_matrix


if __name__ == '__main__':
    args = docopt(__doc__, version='NLI task - version w/ POS and optional sentiment tagging')
    algorithm = args["--algorithm"]
    save_results = args["--save"]
    level_separation = args["--levels"]
    emo = args["--emo"]
    print(emo)

    # Open .csv as dataframe
    data = pd.read_csv('index.csv')
    data = data.drop(columns=['prompt'])

    # Iterate over each level (according to --levels)
    levels = ['low', 'medium', 'high'] if level_separation else ['all']
    for level in levels:
        print(f"Processing {'all data' if level == 'all' else 'level: ' + level}...\n")
        level_data = data[data['level'] == level] if level != 'all' else data

        # Fit the TF-IDF vectorizer to the entire dataset of POS tagged texts
        textlist = []
        pos_tag_list = []
        for name in level_data['name']:
            text = read_file(name)
            textlist.append(text)
            pos_tags = get_pos_tags(text)
            pos_tag_list.append(pos_tags)
        tagged_textlist = [f"{text} {tags}" for text, tags in zip(textlist, pos_tag_list)]

        vectorizer = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 3))
        vectorizer.fit(tagged_textlist)

        # Initialize dataset lists of vectors and classes
        X_list = []
        y_list = []

        # Process files in chunks
        chunk_size = 1000  # Number of files to process at a time
        for start in range(0, len(level_data), chunk_size):
            end = min(start + chunk_size, len(level_data))
            chunk = level_data.iloc[start:end]

            chunk_texts = []
            chunk_pos_tags = []
            chunk_classes = []
            if emo == True: chunk_sentiments = []

            for name, lang in zip(chunk['name'], chunk['L1']):
                text = read_file(name)
                chunk_texts.append(text)

                pos_tags = get_pos_tags(text)
                chunk_pos_tags.append(pos_tags)

                chunk_classes.append(lang)

                if emo == True: sentiment = get_sentiment(text); chunk_sentiments.append(sentiment)

            # Transform the chunk of tagged texts to a chunk of vectors
            tagged_chunk_texts = [f"{text} {tags}" for text, tags in zip(chunk_texts, chunk_pos_tags)]
            X_chunk = vectorizer.transform(chunk_texts)
            
            # Convert sentiment scores to a sparse matrix and append them as a new feature column
            if emo == True:
                sentiment_array = np.array(chunk_sentiments).reshape(-1, 1)
                X_sentiment_chunk = csr_matrix(sentiment_array)  # Stack sentiment as a sparse matrix
            
                # Combine TF-IDF features with sentiment features
                X_chunk = hstack([X_chunk, X_sentiment_chunk])
            y_chunk = np.array(chunk_classes)

            # Append each chunk to the corresponding list
            X_list.append(X_chunk)
            y_list.append(y_chunk)

        # Combine lists into a sparse matrix and label array for the SVM
        X = vstack(X_list)
        print(X.get_shape())
        y = np.concatenate(y_list)

        # Split the dataset into 80% training and 20% test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        # Create and train the chosen classifier on the training set
        classifier = choose_method(algorithm)
        classifier.fit(X_train, y_train)

        # Make predictions for the test set
        y_pred = classifier.predict(X_test)

        # Evaluate the classifier and save classification report in results folder
        print(classification_report(y_test, y_pred, zero_division = 0))
        

        # Display and save confusion matrix (if -m)
        if save_results:
            e = '_no_emo' if emo == False else ''
            combination = algorithm + '_' + level + e
            save_report(y_test, y_pred, combination)
            show_and_save_matrix(y_test, y_pred, classifier, combination)
