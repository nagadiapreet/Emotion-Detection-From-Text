import nltk
import pandas as pd
import tensorflow as tf
import numpy as np
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import re
from tensorflow.keras import regularizers
from tensorflow import keras
import h5py
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import os
from tkinter import font as tkFont
import webbrowser
import random
from nltk.corpus import stopwords
from textblob import Word
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

def showEmoji(msg):
	index = random.choice([0,1,2,3,4])
	if msg == 1:
		panel.configure(image = SadEmotionImage, bg="black")
		url.set(sadLinks[index])
	else:
		panel.configure(image = HappyEmotionImage, bg="black")
		url.set(happyLinks[index])
	print(str(url.get()))
	panel.place(x=330, y=150)

def finalFunction():
	data = pd.read_csv('text_emotion.csv')

	data = data.drop('author', axis=1)

	# Dropping rows with other emotion labels
	data = data.drop(data[data.sentiment == 'anger'].index)
	data = data.drop(data[data.sentiment == 'boredom'].index)
	data = data.drop(data[data.sentiment == 'enthusiasm'].index)
	data = data.drop(data[data.sentiment == 'empty'].index)
	data = data.drop(data[data.sentiment == 'fun'].index)
	data = data.drop(data[data.sentiment == 'relief'].index)
	data = data.drop(data[data.sentiment == 'surprise'].index)
	data = data.drop(data[data.sentiment == 'love'].index)
	data = data.drop(data[data.sentiment == 'hate'].index)
	data = data.drop(data[data.sentiment == 'neutral'].index)
	data = data.drop(data[data.sentiment == 'worry'].index)

	# Making all letters lowercase
	data['content'] = data['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))

	# Removing Punctuation, Symbols
	data['content'] = data['content'].str.replace('[^\w\s]',' ')

	# Removing Stop Words using NLTK
	stop = stopwords.words('english')
	data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

	#Lemmatisation
	data['content'] = data['content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
	#Correcting Letter Repetitions

	def de_repeat(text):
	    pattern = re.compile(r"(.)\1{2,}")
	    return pattern.sub(r"\1\1", text)

	data['content'] = data['content'].apply(lambda x: " ".join(de_repeat(x) for x in x.split()))

	# Code to find the top 10,000 rarest words appearing in the data
	freq = pd.Series(' '.join(data['content']).split()).value_counts()[-10000:]

	# Removing all those rarely appearing words from the data
	freq = list(freq.index)
	data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

	#Encoding output labels 'sadness' as '1' & 'happiness' as '0'
	lbl_enc = preprocessing.LabelEncoder()
	y = lbl_enc.fit_transform(data.sentiment.values)

	# Splitting into training and testing data in 90:10 ratio
	X_train, X_val, y_train, y_val = train_test_split(data.content.values, y, stratify=y, random_state=42, test_size=0.1, shuffle=True)

	# Extracting TF-IDF parameters
	tfidf = TfidfVectorizer(max_features=1000, analyzer='word',ngram_range=(1,3))
	X_train_tfidf = tfidf.fit_transform(X_train)
	X_val_tfidf = tfidf.fit_transform(X_val)

	# Extracting Count Vectors Parameters
	count_vect = CountVectorizer(analyzer='word')
	count_vect.fit(data['content'])
	X_train_count =  count_vect.transform(X_train)
	X_val_count =  count_vect.transform(X_val)
	
	sentence = str(InputText.get())
	print(sentence)
	tweets = pd.DataFrame([sentence])

	# Doing some preprocessing on these tweets as done before
	tweets[0] = tweets[0].str.replace('[^\w\s]',' ')
	stop = stopwords.words('english')
	tweets[0] = tweets[0].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
	tweets[0] = tweets[0].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

	tweet_count = count_vect.transform(tweets[0])

	lsvm = joblib.load('emotions.pkl')  
	tweet_pred = lsvm.predict(tweet_count)
	print(tweet_pred)

	showEmoji(tweet_pred)

def CreateSuggestion():
	print("Called Suggestion")
	print(str(url.get()))
	if not str(url.get())=="nothing":
		webbrowser.open(str(url.get()),new=1)
	else:
		print("Please Enter Message and press Button to check your Emotion.")

master = tk.Tk()
master.geometry("800x500")

url = StringVar()
url.set("nothing")

InputTextLabel = Label(master, text = "Enter Message: ", bg="black", fg="white").place(x = 30,y = 25)  
InputText = Entry(master, width="70")
InputText.place(x = 150, y = 25)  
checkEmotion = Button(master, text = "Check Emotion",activebackground = "black", activeforeground = "white", bg="white", command= finalFunction).place(x = 330, y = 65)

panel = tk.Label(master)
SadEmotionImage = Image.open("sad.png")
SadEmotionImage = SadEmotionImage.resize((150, 150), Image.BILINEAR)
SadEmotionImage = ImageTk.PhotoImage(SadEmotionImage)

HappyEmotionImage = Image.open("happy.png")
HappyEmotionImage = HappyEmotionImage.resize((150, 150), Image.BILINEAR)
HappyEmotionImage = ImageTk.PhotoImage(HappyEmotionImage)

Suggestion = Button(master, text="Our Suggestion", activebackground = "black", activeforeground = "white", command= CreateSuggestion, bg="white", fg="black")
Suggestion.place(x=325, y=350)

happyLinks = ['https://www.youtube.com/watch?v=E_g263MKjhY','https://www.youtube.com/watch?v=1dpX1j_OPyg','https://www.youtube.com/watch?v=eszAgtg-Z9U','https://www.youtube.com/watch?v=v5R75oeyQFY','https://www.youtube.com/watch?v=Y5Wt-FBLGWk']
sadLinks = ['https://www.youtube.com/watch?v=eAK14VoY7C0','https://www.youtube.com/watch?v=rGpwRlCOLbY','https://www.youtube.com/watch?v=PgluKyEwuoE','https://www.youtube.com/watch?v=HHgQO_rKLHU','https://www.youtube.com/watch?v=gORVxUPTez0']

master.title("AI Therapist")
master.configure(background='black')

master.mainloop()