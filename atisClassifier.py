import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import re
import string

data = pd.read_csv('atis_intents.csv', names=['intent','content'])
data_train = pd.read_csv('atis_intents_train.csv',names=['intent', 'content'])
data_test =  pd.read_csv('atis_intents_test.csv', names=['intent', 'content'])

# Pre-processing and stemming
from nltk.corpus import stopwords
nltk.download('stopwords')
stemmer = nltk.stem.SnowballStemmer('english')
sw = stopwords.words('english')

numPunctLowerStem = lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in sw]).lower()

data_train['content_stemmed'] = data_train['content'].apply(numPunctLowerStem)
data_test['content_stemmed'] = data_test['content'].apply(numPunctLowerStem)


tfidf = TfidfVectorizer(ngram_range=(1,2), stop_words='english', sublinear_tf=True)

tfidf_train = tfidf.fit_transform(data_train['content_stemmed'].to_list())

# Creating matrix with all training data
matrix_train = pd.DataFrame(data=tfidf_train.toarray(), columns=tfidf.get_feature_names())
matrix_train['intent'] = data_train['intent'].copy()

# Splitting into X and Y data    
feature_cols = matrix_train.columns[:-1]
X_train = matrix_train[feature_cols]
Y_train = matrix_train['intent']


tfidf_test = tfidf.transform(data_test['content_stemmed'].to_list())

# Creating matrix with all test data
matrix_test = pd.DataFrame(data=tfidf_test.toarray(), columns=tfidf.get_feature_names())
matrix_test['intent'] = data_test['intent'].copy()

# Splitting into X and Y data
feature_cols = matrix_test.columns[:-1]
X_test = matrix_test[feature_cols]
Y_test = matrix_test['intent']

svc = LinearSVC(C=1)

svc_model = svc.fit(X_train, Y_train)
accuracy = svc_model.score(X_test, Y_test)

def plot_metrics(x_test, y_test):
    from matplotlib import pyplot
    from sklearn.metrics import plot_confusion_matrix

    plot_confusion_matrix(svc, x_test, y_test, xticks_rotation= 'vertical', normalize='true')
    return pyplot.show()


def predictIntent(entry):
    '''
    entry : String
    '''
    if entry == "" or entry == " ":
        return 'Empty entry. Try again with a valid entry!'
    else:
        entry_Transformed = tfidf.transform([entry])
        predText = " ".join(svc_model.predict(entry_Transformed)[0].split("_"))
        output = "I guess you're talking about " + predText
        return output


