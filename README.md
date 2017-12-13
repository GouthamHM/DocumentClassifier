# DocumentClassifier
A Naive Bayes binary classifier to Classify Ad Sentences as Accepted(1) or Rejected(0)
1. Dataprocessing.py : Takes data from data/farm-ads.txt and creates a data/farm.csv file in which, each  row  of  the  matrix
                       represents a word (i.e., feature), and each column represents an ad (i.e., instance). The value
                       of a certain matrix entry is the count of the word in the corresponding ad.

2. naivebayes.py : A K-fold validation from 2 to 9 is conducted and the plot of accuracy vs folds in saved in 
                   ./captures/FarmDataClassification.png file
