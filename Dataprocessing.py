from sklearn.feature_extraction.text import CountVectorizer
import csv


def vectorize(file):
    fileobj = open(file, "r")
    lines = fileobj.readlines()
    vectorizer = CountVectorizer(input='content', token_pattern=u'(?u)\\b\\w+\\b')
    dataMatrix = vectorizer.fit_transform(lines)
    word_list = vectorizer.get_feature_names()
    dataMatrix = dataMatrix.toarray()
    f = open('./data/farm.csv', 'w')
    writer = csv.writer(f)

    for index,row in enumerate(dataMatrix.T):
        newRow = []
        newRow.append(word_list[index])
        newRow.extend(row)
        writer.writerow(newRow)
    f.close()

if __name__ =='__main__':
    vectorize('./data/farm-ads.txt')



