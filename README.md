                                                        
# Native Language Identification
Native Language Identification is the task of ...

## Dependencies
The only requirements are docopt, scikit-learn, and matplotlib which are included in ``requirements.txt``.
```
pip install requirements.txt
```

## Arguments
The main file ``nli_project.py`` can take different arguments when called from the terminal.
```
python3 -W ignore plsr_regression.py ((-a | --algorithm) (KN | RF | SV)) [-m | --matrix]
```
where
- (-a | --algorithm) (KN | RF | SV) needs to be one of the three possible methods available, which are
  - KNC: k-Neighbors classifier with k = 10
  - RF: Random Forest
  - SVC: Linear Support Vector Classifier
- [-m | --matrix] shows and saves the confusion matrix as a .png file after evaluating the classifier. 
- [-h | --help] shows the help screen.
- --version show the version of the code.

## The data
The data used consists of 12,100 .txt files and one .csv file, both from the [TOEFL11 dataset](https://www.ets.org/research/policy_research_reports/publications/report/2013/jrkv.html) by Blanchard et al. (2013). 
- The text files are TOEFL short essays written by English students and can be found in the data folder. 
- The .csv file relates these text files with some data of the author of each essay, i.e. the prompt given to the student, their level, and most importantly, their native language. 

| name | prompt | L1 | level |
| :---- | :---: | :---: | :--- |
| 88.txt | P6 | KOR | high |
| 278.txt | P6 | DEU | medium |
| 348.txt | P1 | TUR | high |
| 666.txt | P2 | ZHO | medium |
| 733.txt | P6 | TEL | medium |
| ... | ... | ... | ... |

Due to the nature of the task at hand, we will be focusing only on the name of the files and the native language of the author.

## Three possible methods to choose from
This program allows the user to choose between three possible methods of multiclass classification available in the scikit-learn library. The aim is to try all of them so we can compare the given results. 
- [k-Neighbors Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#).
- [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#).
- [Linear Support Vector Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#).

## Evaluation and results
This program also creates a results directory where it stores classification reports and confusion matrices. 
- A file called report_[CHOSEN ALGORITHM].csv will be saved in the results directory after evaluating the model. 
- If (-m | --matrix) is included in the command when running the code, a file containing a confusion matrix will be shown to the user and saved as confusion_matrix_[CHOSEN ALGORITHM].png.

This allows the user to quickly see and compare the results given by the three possible methods.