#!/usr/bin/env python
# coding: utf-8

# In[79]:


import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer 
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import  MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import LogisticRegression


# In[80]:


df = pd.read_csv (r'Reviews.csv')
df['Summary']


# In[81]:


#df=df[:10000]


# In[82]:


df["Sentiment"] = df["Score"].apply(lambda score: 1 if score > 3 else 0)
df.head(5)


# In[83]:


df["Sentiment"].value_counts()


# # This is a balanced dataset. 

# In[ ]:





# # Stopwords removal

# In[82]:


# Obtain additional stopwords from nltk
stop_words = stopwords.words('english')
stop_words.extend(['the', 'a', 'me'])


# Remove stopwords by gensim and nltk and remove words with 2 or less characters.
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)
            
    return result


# In[83]:


'not' in stop_words


# In[79]:


df['clean'] = df['Text'].apply(preprocess)


# In[80]:


df['clean']


# # we will not use stopwords removal. 

# # Stemming

# a better version of the Porter Stemmer since some issues of it were fixed in this stemmer.

# In[84]:


from nltk.stem.snowball import SnowballStemmer
import re
stemmer = SnowballStemmer(language="english")


# In[85]:


def process(s):
    s=str(s)
    s=s.lower()
    s=re.sub(r'[?|!|.|,|)|(|\|/]',r' ',s) # replace these punctuation with space
    token=s.split()
    out=[]
    for w in token:
        print(w)
        out.append(stemmer.stem(w))
    print(out)
    s = " ".join(out)
    s = re.sub(r'[\'|"|#]', r'',s) 
    return s


# In[86]:


df["Summary_Clean"] = df["Summary"].apply(process)


# sent = []
# for row in df['clean']:
#     sequ = ''
#     for word in row:
#         sequ = sequ + ' ' + word
#     sent.append(sequ)
# 
# final_X = sent
# print(final_X[1])

# # Train test split

# In[87]:


train,test = train_test_split(df, test_size=0.2,random_state=1)


# # Encoding by unigram

# In[89]:


vectorizer  = CountVectorizer(ngram_range=(1,1))
bow_data = vectorizer.fit_transform(train['Summary_Clean'])


# In[90]:


train_summary=bow_data


# In[93]:


test_summary = vectorizer.transform(test['Summary_Clean'])


# In[95]:


bow_data.shape


# 24732 有多少词
# 454763 多少review

# # Bigram 

# In[96]:


Bi_vectorizer  = CountVectorizer(ngram_range=(1,2), min_df = 7)
bi_gram_vectors_train = Bi_vectorizer.fit_transform(train['Summary_Clean'].values)
bi_gram_vectors_test = Bi_vectorizer.transform(test['Summary_Clean'].values)


# In[97]:


bi_gram_vectors_train.shape


# In[143]:


bi_gram_vectors_test.shape


# In[98]:


features = Bi_vectorizer.get_feature_names()
features[-20:]


# # Logistic regression model with bigram

# In[99]:


#basic logistic regression 
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression


# In[100]:


lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=0)


# In[101]:


logreg_bi_gram_result = lr.fit(bi_gram_vectors_train, train['Sentiment'])


# In[102]:


prediction = dict()


# In[103]:


prediction['logistic_bi_gram']=lr.predict(bi_gram_vectors_test)


# In[104]:


prediction


# In[105]:


import collections
print('test data')
print(test['Sentiment'].value_counts())
print('--------------')
print('predicted data')
print(collections.Counter(prediction['logistic_bi_gram']))


# In[106]:


prob = dict()
prob['logistic_bi_gram'] = lr.predict_proba(bi_gram_vectors_test)


# In[107]:


prob['logistic_bi_gram'][:,1]


# In[134]:


feature = Bi_vectorizer.get_feature_names()
feature_coefs = pd.DataFrame(
    data = list(zip(feature, lr.coef_[0])),
    columns = ['feature', 'coef'])

feature_coefs.sort_values(by='coef')


# # Logistic regression model with unigram

# In[108]:


lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=0)
logreg_uni_gram_result = lr.fit(train_summary, train['Sentiment'])


# In[109]:


prediction['logistic_uni_gram']=lr.predict(test_summary)


# In[111]:


import collections
print('test data')
print(test['Sentiment'].value_counts())
print('--------------')
print('predicted data')
print(collections.Counter(prediction['logistic_uni_gram']))


# In[112]:


prob['logistic_uni_gram'] = lr.predict_proba(test_summary)


# In[113]:


prob['logistic_uni_gram'][:,1]


# In[114]:


prediction


# In[ ]:





# # Random Forest with bigram

# In[122]:


from sklearn.ensemble import RandomForestClassifier


# In[123]:


rf_bi_gram = RandomForestClassifier(n_estimators = 100, class_weight = 'balanced', n_jobs = -1)
rf_bi_gram_result = rf_bi_gram.fit(bi_gram_vectors_train, train['Sentiment'])


# In[124]:


prediction['rf_bi_gram'] = rf_bi_gram.predict(bi_gram_vectors_test)


# In[125]:


print('test data')
print(test['Sentiment'].value_counts())
print('--------------')
print('predicted data')
print(collections.Counter(prediction['rf_bi_gram']))


# In[126]:


prob['rf_bi_gram'] = rf_bi_gram.predict_proba(bi_gram_vectors_test)


# In[138]:


a=rf_bi_gram.feature_importances_
a.shape


# In[140]:


len(feature)#unique bi grams in the entire dataset


# In[142]:


prediction['rf_bi_gram'] .shape


# # feature importance

# In[136]:


feature = Bi_vectorizer.get_feature_names()
rf_feature_importance = pd.DataFrame(data = list(zip(feature, rf_bi_gram.feature_importances_)),
    columns = ['feature', 'importance'])
rf_feature_importance.sort_values(by='importance', ascending=False)


# # Result

# In[127]:


from sklearn import metrics
from sklearn.metrics import roc_curve, auc
cmp = 0
colors = ['b', 'g', 'y', 'm', 'k']
for model, predicted in prediction.items():
    false_positive_rate, true_positive_rate, thresholds = roc_curve(test['Sentiment'].values, prob[model][:,1])
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f'% (model,roc_auc))
    cmp += 1

plt.title('encoding methods comparaison with ROC')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[128]:


print(metrics.classification_report(test['Sentiment'].values, 
                                    prediction['logistic_bi_gram'], 
                                    target_names = ['0', '1']))


# In[129]:


print(metrics.classification_report(test['Sentiment'].values, 
                                    prediction['logistic_uni_gram'], 
                                    target_names = ['0', '1']))


# In[130]:


print(metrics.classification_report(test['Sentiment'].values, 
                                    prediction['rf_bi_gram'], 
                                    target_names = ['0', '1']))


# # save as pickle

# In[146]:


import pickle
pickle.dump(lr, open('amazon.lr.pickle', 'wb'))
pickle.dump(rf_bi_gram, open('amazon.rf.pickle', 'wb'))    


# In[148]:


logreg_bi_gram = pickle.load(open('amazon.lr.pickle', 'rb'))
rf_bi_gram = pickle.load(open('amazon.rf.pickle', 'rb'))


# In[ ]:




