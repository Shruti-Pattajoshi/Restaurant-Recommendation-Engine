import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer


df = pd.read_csv('dataset_food_online.txt', encoding = "ISO-8859-1")

#Checks for the NULL values in Dataframe
df.isnull().sum()

#printing the first few rows of the DataFrame
df.head()


#cleaning the textual data:
import string
import re


def clean_text(text):
    text = text.translate(string.punctuation)

    ## Convert words to lower case and split them
    text = text.lower().split()

    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


yelp_data = df[['business_id', 'user_id', 'stars', 'text']]

yelp_data['text'] = yelp_data['text'].apply(clean_text)

#splitting training and testing datasets
vld_size=0.15
X_train, X_valid, y_train, y_valid = train_test_split(yelp_data['text'], df['business_id'], test_size = vld_size)

userid_df = yelp_data[['user_id','text']]
business_df = yelp_data[['business_id', 'text']]


userid_df = userid_df.groupby('user_id').agg({'text': ' '.join})
business_df = business_df.groupby('business_id').agg({'text': ' '.join})

#userid vectorizer
userid_vectorizer = TfidfVectorizer(tokenizer = WordPunctTokenizer().tokenize, max_features=1000)
userid_vectors = userid_vectorizer.fit_transform(userid_df['text'])
#userid_vectors.shape

#Business id vectorizer
businessid_vectorizer = TfidfVectorizer(tokenizer = WordPunctTokenizer().tokenize, max_features=1000)
businessid_vectors = businessid_vectorizer.fit_transform(business_df['text'])
#businessid_vectors.shape

#Matrix Factorization ::
userid_rating_matrix = pd.pivot_table(yelp_data, values='stars', index=['user_id'], columns=['business_id'])
#userid_rating_matrix.shape


#userid_rating_matrix.head()

P = pd.DataFrame(userid_vectors.toarray(), index=userid_df.index, columns=userid_vectorizer.get_feature_names())
Q = pd.DataFrame(businessid_vectors.toarray(), index=business_df.index, columns=businessid_vectorizer.get_feature_names())


def matrix_factorization(R, P, Q, steps=1, gamma=0.001, lamda=0.02):
    for step in range(steps):
        for i in R.index:
            for j in R.columns:
                if R.loc[i, j] > 0:
                    eij = R.loc[i, j] - np.dot(P.loc[i], Q.loc[j])
                    P.loc[i] = P.loc[i] + gamma * (eij * Q.loc[j] - lamda * P.loc[i])
                    Q.loc[j] = Q.loc[j] + gamma * (eij * P.loc[i] - lamda * Q.loc[j])
        e = 0
        for i in R.index:
            for j in R.columns:
                if R.loc[i, j] > 0:
                    e = e + pow(R.loc[i, j] - np.dot(P.loc[i], Q.loc[j]), 2) + lamda * (
                                pow(np.linalg.norm(P.loc[i]), 2) + pow(np.linalg.norm(Q.loc[j]), 2))
        if e < 0.001:
            break

    return P, Q

P, Q = matrix_factorization(userid_rating_matrix, P, Q, steps=1, gamma=0.001,lamda=0.02)



#Storing P, Q and vectorizer in pickle file
import pickle
output = open('recommendation.pkl', 'wb')
pickle.dump(P,output)
pickle.dump(Q,output)
pickle.dump(userid_vectorizer,output)
output.close()



words = 'I am interested in vegetarian restaurants and I want eat sushi'
test_df= pd.DataFrame([words], columns=['text'])
test_df['text'] = test_df['text'].apply(clean_text)
test_vectors = userid_vectorizer.transform(test_df['text'])
test_v_df = pd.DataFrame(test_vectors.toarray(), index=test_df.index, columns=userid_vectorizer.get_feature_names())

predictItemRating=pd.DataFrame(np.dot(test_v_df.loc[0],Q.T),index=Q.index,columns=['Rating'])
topRecommendations=pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending=[0])[:3]
topRecommendations
#printing top recommendations

f = open('recommendation.pkl', 'rb')
P, Q, userid_vectorizer = pickle.load(f), pickle.load(f), pickle.load(f)


test_df = pd.DataFrame([sentence], columns=['text'])
test_df['text'] = test_df['text'].apply(self.clean_text)
test_vectors = userid_vectorizer.transform(test_df['text'])
test_v_df = pd.DataFrame(test_vectors.toarray(), index=test_df.index,
                         columns=userid_vectorizer.get_feature_names())


test_df = X_valid.to_frame()
test_df['text'] = test_df['text'].apply(clean_text)
test_vectors = userid_vectorizer.transform(test_df['text'])
test_v_df = pd.DataFrame(test_vectors.toarray(), index=test_df.index,
                         columns=userid_vectorizer.get_feature_names())


y_pred = []
for key, row in test_v_df.iterrows():
    predictItemRating=pd.DataFrame(np.dot(row,Q.T),index=Q.index,columns=['Rating'])
    topRecommendations=pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending=[0])[:1]
    y_pred.append(topRecommendations.index[0])


#Calculating Accuracy
from sklearn.metrics import accuracy_score
print('Accuracy for validation set is: ',accuracy_score(y_valid, y_pred))

# Accuracy obtained is : 67.6 %


