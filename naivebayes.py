from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# To find the prior prob of Training Data
def prior_probability(list_of_labels):
    unique, counts = np.unique(list_of_labels, return_counts=True)
    pos_neg = dict(zip(unique, counts))
    len_of_labels = len(list_of_labels)
    return_val = {'pos': 0, 'neg': 0}

    if 0 in pos_neg:
        return_val['neg'] = np.log10(float(pos_neg[0]) / len_of_labels)
    if 1 in pos_neg:
        return_val['pos'] = np.log10(float(pos_neg[1]) / len_of_labels)

    return return_val


# Get total counts
def total_count_in_each_class(word):
    pos_neg = np.sum(word, axis=0)
    print pos_neg
    return {'pos': pos_neg[0], 'neg': pos_neg[1]}


# Get loglikelihood of each word
def loglikelihood(word, total_pos, total_neg, vocal_len):
    print vocal_len
    for row in word:
        row[0] = np.log10(float(row[0] + 1) / (total_pos + vocal_len))
        row[1] = np.log10(float(row[1] + 1) / (total_neg + vocal_len))


def getdata(filename):
    fileobj = open(filename, "r")
    lines = fileobj.readlines()
    fileobj.close()
    # vectorizer = CountVectorizer(input='content', token_pattern=u'(?u)\\b\\w+\\b')
    # dataMatrix = vectorizer.fit_transform(lines)
    # dataMatrix = dataMatrix.toarray()

    return lines


def getlabels(filename):
    return np.loadtxt(filename, dtype=np.int8, usecols=(1,))


# To Train Naive bayes
def naiveBayes(fileName, labelfileName, portion):
    lines = np.array(getdata(fileName))
    list_of_labels = getlabels(labelfileName)

    X_train, X_test, y_train, y_test = train_test_split(lines,
                                                        list_of_labels,
                                                        train_size=portion,
                                                        shuffle=True)
    vectorizer = CountVectorizer(input='content', token_pattern=u'(?u)\\b\\w+\\b')
    X_train = vectorizer.fit_transform(X_train)
    X_train = X_train.toarray()
    word_list = vectorizer.get_feature_names()
    word = np.zeros(shape=[X_train.shape[1], 2])

    for index1, column in enumerate(X_train.T):
        for index, ad in enumerate(column):
            if y_train[index] == 1:
                word[index1][0] += ad
            else:
                word[index1][1] += ad

    pos_neg_counts = total_count_in_each_class(word)
    prior = prior_probability(y_train)
    loglikelihood(word, pos_neg_counts['pos'], pos_neg_counts['neg'], len(word_list))

    correct_preds = 0
    for index_ads, ads in enumerate(X_test):
        pos = prior['pos']
        neg = prior['neg']
        for index, word_in in enumerate(ads.split(" ")):
            # Find ind which word exists in the ad
            if word_in in word_list:
                new_index = word_list.index(word_in)
                pos += word[new_index][0]
                neg += word[new_index][1]

        if pos > neg and y_test[index_ads] == 1:
            correct_preds += 1
        elif neg > pos and y_test[index_ads] == 0:
            correct_preds += 1
    return float(correct_preds) / len(y_test)


if __name__ == '__main__':
    list_of_portions = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
    naive_bayes_accuracy = [0] * len(list_of_portions)

    for index, portion in enumerate(list_of_portions):
        sum = 0
        for i in range(5):
            sum += naiveBayes('./data/farm-ads.txt', './data/farm-ads-label.txt', portion)
        naive_bayes_accuracy[index] = sum/5
    plt.plot( list_of_portions,[x*100 for x in naive_bayes_accuracy])
    plt.xlabel('Test Portion considered')
    plt.ylabel('Accuracy')
    plt.savefig('./captures/FarmDataClassification.png')

