import gc
import re
import nltk
import pandas
import xgboost

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


stops = set(stopwords.words('english'))


def clean_message(msg):
    word_tokens = re.split(r'\W+', msg)
    filtered_tokens = [w for w in word_tokens if not w in stops]

    return ' '.join(filtered_tokens)
# end def clean_message


def number_channels(chnl):
    number = {
        "games": 0,
        "stonks": 1,
        'general':2, 'dj-booth':3, 'memes':4, 'toms-memes':5, 'gluten':6,
 'work':7, 'duckbutt':8
    }

    return number[chnl]
# end def number_channels


# Load Primer Data
df1 = pandas.read_csv("primer.csv", delimiter="|")
df1 = df1[["channel","message"]]

# Load Historical Data
df2 = pandas.read_csv("clean_messages.csv", delimiter="|")
df2 = df2[~(df2["message"].str.contains("<@"))]
df2 = df2[["channel","message"]]

# Merge Primer to History
df = pandas.concat([df1, df2])
del df1
del df2
gc.collect()

print({
        "games": 0,
        "stonks": 1,
        'general':2, 'dj-booth':3, 'memes':4, 'toms-memes':5, 'gluten':6,
 'work':7, 'duckbutt':8
    })
df["channel"] = df["channel"].map(number_channels)

# Clean Messages
df["message"] = df["message"].map(clean_message)

# Build TF-IFD Vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(df["message"].values)

# Build Train & Test Sets
x = df["message"].values
x = vectorizer.transform(x)

y = df["channel"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

# Train Model
clf = GaussianNB()

clf.fit(x_train.toarray(),y_train)
a = clf.score(x_test.toarray(),y_test)
print(a)
b = clf.predict(x_test.toarray())

from sklearn.metrics import confusion_matrix,f1_score, precision_score,recall_score
c = confusion_matrix(y_test, b)
print(c)


model = xgboost.XGBRegressor(objective ='multi:softprob', num_class=9)
model.fit(x_train,y_train)
preds = model.predict(x_test)
print()
acts = [1 if x == 0 else 0 for x in y_test]
#print(preds[:,0])
#print(acts)

from sklearn.metrics import roc_auc_score, roc_curve
roc = roc_auc_score(acts, preds[:,0])
print(roc)

import numpy
fpr, tpr, threshold = roc_curve(acts, preds[:,0])
optimal_idx = numpy.argmax(tpr - fpr)
optimal_threshold = threshold[optimal_idx]
print(optimal_threshold)


new_t = vectorizer.transform([clean_message("guys diablo 2 remastered is getting announced would anyone be down for it")])
new_p = model.predict(new_t)
print(new_p)

if new_p[0][0] >= 0.21181695:
    print("IT IS A GAME MESSAGE!!!!!")
else:
    print(list(new_p[0]))
