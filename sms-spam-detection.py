#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('spam.csv',encoding="latin-1")


# In[3]:


df.sample(5)


# In[4]:


df.shape


# ## 1. Data Cleaning

# In[5]:


df.info()


# In[6]:


# drop cols
df.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],inplace = True)


# In[7]:


df.sample(5)


# In[8]:


# rename cols
df.rename(columns = {'v1':'Target','v2':'Text'},inplace = True)
df.sample(5)


# In[9]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[10]:


df['Target'] = encoder.fit_transform(df['Target'])


# In[11]:


df.head()


# In[12]:


# check missing values
df.isnull().sum()


# In[13]:


# check for duplicate values
df.duplicated().sum()


# In[14]:


#remove duplicate
df = df.drop_duplicates(keep='first')


# In[15]:


df.duplicated().sum()


# In[16]:


df.shape


# ## 2. EDA

# In[17]:


df.head()


# In[18]:


df['Target'].value_counts()


# In[19]:


import matplotlib.pyplot as plt
plt.pie(df['Target'].value_counts(),labels=['ham','spam'],autopct='%0.2f')
plt.show()


# In[20]:


# Data is imbalanced


# In[21]:


get_ipython().system('pip install nltk')


# In[22]:


import nltk


# In[23]:


nltk.download('punkt')


# In[24]:


# num of characters
df['num_characters'] = df['Text'].apply(len)


# In[25]:


df.head()


# In[26]:


# num of words
df['num_words'] = df['Text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[27]:


df.head()


# In[28]:


# num of sentences
df['num_sentences'] = df['Text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[29]:


df.head()


# In[31]:


df[['num_characters','num_words','num_sentences']].describe()


# In[32]:


# ham
df[df['Target'] == 0][['num_characters','num_words','num_sentences']].describe()


# In[44]:


# spam
df[df['Target'] == 1][['num_characters','num_words','num_sentences']].describe()


# In[45]:


import seaborn as sns


# In[46]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['Target'] == 0]['num_characters'])
sns.histplot(df[df['Target'] == 1]['num_characters'],color='red')


# In[47]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['Target'] == 0]['num_words'])
sns.histplot(df[df['Target'] == 1]['num_words'],color='red')


# In[48]:


sns.pairplot(df,hue='Target')


# In[49]:


sns.heatmap(df.corr(),annot=True)


# ## 3. Data Preprocessing
# - Lower case
# - Tokenization
# - Removing special characters
# - Removing stop words and punctuation
# - Stemming

# In[74]:


from nltk.corpus import stopwords
import string


# In[75]:


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()
            
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)


# In[76]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('dancing')


# In[77]:


transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.")


# In[78]:


df['Transformed_text'] = df['Text'].apply(transform_text)


# In[79]:


df.head()


# In[101]:


get_ipython().system('pip install wordcloud')


# In[80]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[81]:


spam_wc = wc.generate(df[df['Target'] == 1]['Transformed_text'].str.cat(sep=" "))


# In[82]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[83]:


ham_wc = wc.generate(df[df['Target'] == 0]['Transformed_text'].str.cat(sep=" "))


# In[84]:


plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[85]:


spam_corpus = []
for msg in df[df['Target'] == 1]['Transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[86]:


len(spam_corpus)


# ## 4.Model Building

# In[124]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)


# In[125]:


X = tfidf.fit_transform(df['Transformed_text']).toarray()


# In[126]:


X.shape


# In[127]:


X


# In[128]:


y = df['Target'].values


# In[129]:


y


# In[130]:


from sklearn.model_selection import train_test_split


# In[131]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[132]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[133]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[134]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[135]:


mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[136]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[137]:


#tfid --> MNB


# In[138]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[139]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)


# In[140]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc,  
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb,
}


# In[141]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[144]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[145]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[146]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


# In[123]:


performance_df


# In[147]:


# model improve
# 1. Change the max_features parameter of TfIdf


# In[148]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_score})


# In[149]:


#creating pipeline
import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[ ]:




