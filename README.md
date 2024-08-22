                                                        
# Native Language Identification
Native Language Identification is the task of identifying an individual’s first language (L1) based on the patterns they show when using another one (L2). The underlying assumption behind this is that the native language of a writer predisposes them to exhibit some specific linguistic patterns or quirks when writing in a second language, so we can use this cross-linguistic inference or native language interference to infer a writer's L1 from their use of their L2.

## Dependencies
The only requirements are docopt, scikit-learn, and matplotlib which are included in ``requirements.txt``.
```
pip install requirements.txt
```

## Arguments
The main file ``nli_project.py`` can take different arguments when called from the terminal.
```
python3 -W ignore plsr_regression.py ((-a | --algorithm) (KN | RF | SV)) [-l | --levels] [-s | --save]
```
where
- ``(-a | --algorithm) (KN | RF | SV)`` needs to be one of the three methods available, which are
  - KN: k-Neighbors classifier with k = 10
  - RF: Random Forest classifier
  - SV: Linear Support Vector classifier
- ``[-l | --levels]`` separates the data in three parts according to the English level of the authors of the texts.
- ``[-s | --save]`` shows and saves the confusion matrix as a .png file and the report as a .csv file after evaluating the classifier. 
- ``[-e | --emo]`` includes sentiment analysis in the feature extraction process. It defaults to True.
- ``[-h | --help]`` shows the help screen.
- ``--version`` shows the version of the code.

## The data
The data used consists of 12,100 .txt files and one .csv file, both from the [TOEFL11 dataset](https://www.ets.org/research/policy_research_reports/publications/report/2013/jrkv.html) by Blanchard et al. (2013). 
- The text files are TOEFL short essays written by English students and can be found in the data folder. 
- The .csv file relates these text files with some data of the author of each essay, i.e. the prompt given to the student, their proficiency level, and most importantly, their native language. 

| name | prompt | L1 | level |
| :---- | :---: | :---: | :--- |
| 88.txt | P6 | KOR | high |
| 278.txt | P6 | DEU | medium |
| 348.txt | P1 | TUR | high |
| 666.txt | P2 | ZHO | medium |
| 733.txt | P6 | TEL | medium |
| ... | ... | ... | ... |

Due to the nature of the task at hand, we will be focusing mostly on the name of the files and the native language of the author. However, we can also choose to separate the data according to the level and perform the task for each level by including ``--levels`` in the command line.

## Different methods to choose from
This program allows the user to choose between three possible methods of multiclass classification available in the scikit-learn library. The aim is to try all of them so we can compare the given results. 
- [k-Neighbors Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#)
- [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#)
- [Linear Support Vector Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#)

The user can also choose to separate the data according to the English level of the authors of the texts, and to include sentiment analysis in the feature extraction process. This is useful to compare the results obtained through different methods. 

## Evaluation and results
A classification report will be printed every time the code is ran. 
Also, if ``(-s | --save)`` is included in the command when running the code, results will be saved in the results directory. 
- The aforementioned report will be saved as ``report_[ALGORITHM]_[LEVEL].csv``.
- A confusion matrix will be shown to the user and saved as ``confusion_matrix_[ALGORITHM]_[LEVEL].png``.

This allows the user to quickly see and compare the results given by different methods.