# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 12:55:33 2021

@author: ASUS
"""
# Importing Required Libraries
import re
import nltk
import spacy
import gensim
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import text2emotion as te
from gensim import corpora
from bs4 import BeautifulSoup
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.models import ldamodel
from nltk.probability import FreqDist
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

# Webscraping Amazon Reviews of New Apple iPhone 12 (64GB) - Blue
title_list=[]
comment_list=[]
rating_list=[]
for i in range(1,92):
    url='https://www.amazon.in/New-Apple-iPhone-12-64GB/product-reviews/B08L5WHFT9/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber='+str(i)
    r=requests.get(url)
    soup= BeautifulSoup(r.text,'html.parser')
    reviews=soup.find_all('div',{'data-hook':'review'})
    name=soup.title.text
    for items in reviews:
        title=items.find('a',{'data-hook':'review-title'}).text.strip()
        comments=items.find('span',{'data-hook':'review-body'}).text.strip()
        rating=float(items.find('i',{'data-hook':'review-star-rating'}).text.replace('out of 5 stars','')
                                .strip())
        title_list.append(title)
        comment_list.append(comments)
        rating_list.append(rating)
name=name.replace('Amazon.in:Customer reviews:','')
print(" Product Name : "+name)
data=pd.DataFrame(columns=["Title","Comments","Rating"])
data["Title"]=title_list
data["Comments"]=comment_list
data["Rating"]=rating_list

# Creating Spacy Object
nlp=spacy.load('en_core_web_sm')

# Data Pre-Processing
data["Rating"].value_counts().plot(kind="pie")
Comments=[]
Title=[]
for i in range(0,len(data)):
    raw=re.sub(r'[^a-zA-Z0-9_\s]+',' ',data["Comments"][i])
    raw=re.sub(r'\d',' ',raw)
    raw_title=re.sub(r'[^a-zA-Z0-9_\s]+',' ',data["Title"][i])    
    raw_title=re.sub(r'\d',' ',raw_title)
    Comments.append(nlp(raw.lower()))
    Title.append(nlp(raw_title.lower()))
words_Comments=[]
lemmatized_Comments=[]
for i in Comments:
    words_Comments.append([token for token in i if not token.is_stop and token.is_alpha and len(token)>2])
    lemmatized_Comments.append([token.lemma_ for token in i if not token.is_stop and token.is_alpha and len(token)>2])
words_title=[]
lemmatized_title=[]
for i in Title:
    words_title.append([token for token in i if not token.is_stop and token.is_alpha and len(token)>2])
    lemmatized_title.append([token.lemma_ for token in i if not token.is_stop and token.is_alpha and len(token)>2])
lis_comments=[]
lis_title=[]
for i in lemmatized_Comments:
    l=' '.join([token for token in i])
    lis_comments.append(l)
for i in lemmatized_title:
    t=' '.join([token for token in i])
    lis_title.append(t)

# Entity Recognition
enr_comments=[]
enr_title=[]
for i in range(0,len(data)):
    enr_comments.append(nlp(data["Comments"][i]))
    enr_title.append(nlp(data["Title"][i]))
entity_comments=[]
entity_title=[]
for i in enr_comments:
    entity_comments.append([(m,m.label_) for m in i.ents])
for i in enr_title:
    entity_title.append([(n,n.label_) for n in i.ents])
print(entity_comments)
print(entity_title)

# Word Frequency
fdist_comments=FreqDist()
fdist_title=FreqDist()
for i in lemmatized_Comments:
    for token in i:
        fdist_comments[token]+=1
for i in lemmatized_title:
    for token in i:
        fdist_title[token]+=1
fdist_com_max_words=pd.DataFrame(data=fdist_comments.most_common(15),columns=["Words","Count"])    
fdist_title_max_words=pd.DataFrame(data=fdist_title.most_common(15),columns=["Words","Count"])    
# Visualisation of Most Repeated Words in Comments
plot=sns.barplot(x=fdist_com_max_words["Words"],y=fdist_com_max_words["Count"])
plot.set_xticklabels(plot.get_xticklabels(),rotation=45,horizontalalignment='right')
plt.title("Most Repeated Words in Comments")
# Visualisation of Most Repeated Words in Title
plot2=sns.barplot(x=fdist_title_max_words["Words"],y=fdist_title_max_words["Count"])
plot2.set_xticklabels(plot2.get_xticklabels(),rotation=45,horizontalalignment='right')
plt.title("Most Repeated Words in Title")

# Topic Modelling
# For Comments
doc_com=[doc for doc in lemmatized_Comments]
dictn_com=corpora.Dictionary(doc_com)
dtm_com=[dictn_com.doc2bow(doc) for doc in doc_com]
lda=ldamodel.LdaModel
lda_com_model=lda(dtm_com,num_topics=10,id2word=dictn_com,passes=50)
print(lda_com_model.print_topics()) # Topics generated from Comments
# For Title
doc_title=[doc for doc in lemmatized_title]
dictn_title=corpora.Dictionary(doc_title)
dtm_title=[dictn_title.doc2bow(doc) for doc in doc_title]
lda_title_model=lda(dtm_title,num_topics=10,id2word=dictn_title,passes=50)
print(lda_title_model.print_topics()) # Topics generated from Title

# Tfidf Vectorizer for Comments
cv=TfidfVectorizer()
response_com=cv.fit_transform(lis_comments)
f_vect_com=response_com[0]
vect_com=pd.DataFrame(data=f_vect_com.T.todense(),index=cv.get_feature_names(),columns=["TFIDF_Com_Scores"])
vect_com.sort_values(by=["TFIDF_Com_Scores"],ascending=False)
# Tfidf Vectorizer for Title
response_title=cv.fit_transform(lis_title)
f_vect_title=response_title[0]
vect_title=pd.DataFrame(data=f_vect_title.T.todense(),index=cv.get_feature_names(),columns=["TFIDF_Com_Scores"])
vect_title.sort_values(by=["TFIDF_Com_Scores"],ascending=False)

# Word2Vec for Comments
word2vec_comments=pd.DataFrame(lis_comments,columns=["Comments"])
word2vec_comments=word2vec_comments.Comments.apply(gensim.utils.simple_preprocess)
model_com=Word2Vec(word2vec_comments,min_count=1,max_vocab_size=100000)
model_com.wv.key_to_index  # Comment's Model Vocabulary Check
model_com.corpus_count
model_com.epochs
model_com.wv.most_similar('contact')
model_com.wv.get_vecattr('apple','count') # Check Word Count
len(model_com.wv) # Length of Vocabulary
# Word2Vec for Title
word2vec_title=pd.DataFrame(lis_title,columns=["Title"])
word2vec_title=word2vec_title.Title.apply(gensim.utils.simple_preprocess)
model_title=Word2Vec(word2vec_title,min_count=1,max_vocab_size=100000)
model_title.wv.key_to_index  # Comment's Model Vocabulary Check
model_title.corpus_count
model_title.epochs
model_title.wv.most_similar('charge')
model_title.wv.get_vecattr('apple','count') # Check Word Count
len(model_title.wv) # Length of Vocabulary

# Positive Words in Comments
pos_words=open(r"D:\Data Science Assignments\Python-Assignment\NLP and Text Mining\positive-words.txt").read()
pos_dictn=nltk.word_tokenize(pos_words)
gen_com=' '.join([str(elm) for elm in lis_comments])
gencom_words=nltk.word_tokenize(gen_com)
pos_com_matches=list(set(pos_dictn).intersection(set(gencom_words)))
positive_com=len(pos_com_matches)
# Positive Words in Title
gen_title=' '.join([str(elm) for elm in lis_title])
gentitle_words=nltk.word_tokenize(gen_title)
pos_title_matches=list(set(pos_dictn).intersection(set(gentitle_words)))
positive_title=len(pos_title_matches)

# Negative Words in Comments
neg_words=open(r"D:\Data Science Assignments\Python-Assignment\NLP and Text Mining\negative-words.txt").read()
neg_dictn=nltk.word_tokenize(neg_words)
neg_com_matches=list(set(neg_dictn).intersection(set(gencom_words)))
negative_com=len(neg_com_matches)
# Negative Words in Title
neg_title_matches=list(set(neg_dictn).intersection(set(gentitle_words)))
negative_title=len(neg_title_matches)

# Visualisation of positive and negative words
pos_neg=pd.DataFrame(data=[["Positive_Words_Comments",positive_com],
                           ["Positive_Words_Title",positive_title],
                           ["Negative_Words_Comments",negative_com],
                           ["Negative_Words_Title",negative_title]],columns=["Polarity","Count"])
p=sns.barplot(x=pos_neg["Polarity"],y=pos_neg["Count"])
p.set_xticklabels(p.get_xticklabels(),rotation=45,horizontalalignment='right')
plt.title("Positive and Negative Words in Corpus")

# Wordcloud
wc=WordCloud(height=512,width=512,background_color='black',max_font_size=20,
             min_font_size=6,repeat=True,max_words=10000,random_state=40)
# Positive Comment Words WordCloud
poscom=' '.join([str(elm) for elm in pos_com_matches])
wc.generate(poscom)
plt.axis('off')
plt.imshow(wc,interpolation='bilinear')
plt.show()
# Positive Title Words WordCloud
postitle=' '.join([str(elm) for elm in pos_title_matches])
wc.generate(postitle)
plt.axis('off')
plt.imshow(wc,interpolation='bilinear')
plt.show()
# Negative Comment Words WordCloud
negcom=' '.join([str(elm) for elm in neg_com_matches])
wc.generate(negcom)
plt.axis('off')
plt.imshow(wc,interpolation='bilinear')
plt.show()
# Negative Title Words WordCloud
negtitle=' '.join([str(elm) for elm in neg_title_matches])
wc.generate(negtitle)
plt.axis('off')
plt.imshow(wc,interpolation='bilinear')
plt.show()
# Entire Comment WordCloud
wc.generate(gen_com)
plt.axis('off')
plt.imshow(wc,interpolation='bilinear')
plt.show()
# Entire Title WordCloud
wc.generate(gen_title)
plt.axis('off')
plt.imshow(wc,interpolation='bilinear')
plt.show()

# Sentiment Analysis
positive_comment=[]
positive_title=[]
negative_comment=[]
negative_title=[]
for i in data["Comments"]:
    polar=TextBlob(i).sentiment.polarity
    if polar >0:
        positive_comment.append(i)
    else:
        negative_comment.append(i)
for i in data["Title"]:
    pol=TextBlob(i).sentiment.polarity
    if pol >0:
        positive_title.append(i)
    else:
        negative_title.append(i)
print("Positive Comments Count : {}".format(len(positive_comment)))
print("Negative Comments Count : {}".format(len(negative_comment)))
print("Positive Title Count : {}".format(len(positive_title)))
print("Negative Title Count : {}".format(len(negative_title)))

# Emotion Mining of Comments
emo_com=te.get_emotion(gen_com)
emotion_comment=pd.DataFrame(data=emo_com.items(),columns=["Emotions","Count"])
emotion_comment["Count"]=emotion_comment["Count"]*100

# Emotion Mining of Title
emo_title=te.get_emotion(gen_title)
emotion_title=pd.DataFrame(data=emo_title.items(),columns=["Emotions","Count"])
emotion_title["Count"]=emotion_title["Count"]*100

# Visualisation of Emotions in Comments
sns.barplot(x=emotion_comment["Emotions"],y=emotion_comment["Count"])

# Visualisation of Emotions in Comments
sns.barplot(x=emotion_title["Emotions"],y=emotion_title["Count"])
