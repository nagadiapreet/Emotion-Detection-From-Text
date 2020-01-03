import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import os
from tkinter import font as tkFont
import webbrowser
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def nltk2wn_tag(nltk_tag):
  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:          
    return None

def lemmatize_sentence(sentence):
  nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
  wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
  res_words = []
  for word, tag in wn_tagged:
    if tag is None:            
      res_words.append(word)
    else:
      res_words.append(lemmatizer.lemmatize(word, tag))
  return " ".join(res_words)

# function to print sentiments 
# of the sentence. 
def sentiment_scores(sentence): 

	# Create a SentimentIntensityAnalyzer object. 
	sid_obj = SentimentIntensityAnalyzer() 

	# polarity_scores method of SentimentIntensityAnalyzer 
	# oject gives a sentiment dictionary. 
	# which contains pos, neg, neu, and compound scores. 
	sentiment_dict = sid_obj.polarity_scores(sentence) 
	
	print("Overall sentiment dictionary is : ", sentiment_dict) 
	print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative") 
	print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral") 
	print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive") 

	print("Sentence Overall Rated As", " ") 

	# decide sentiment as positive, negative and neutral 
	if sentiment_dict['compound'] >= 0.05 : 
		return "Positive"

	elif sentiment_dict['compound'] <= - 0.05 : 
		return "Negative" 

	else : 
		return "Neutral" 

def showEmoji(msg):
	index = random.choice([0,1,2,3,4])
	if msg == "Negative":
		panel.configure(image = SadEmotionImage, bg="black")
		url.set(sadLinks[index])
	elif msg == "Positive":
		panel.configure(image = HappyEmotionImage, bg="black")
		url.set(happyLinks[index]) 
	else:
		panel.configure(image = NeutralEmotionImage, bg="black")
		url.set(neutralLinks[index])
	print(str(url.get()))
	panel.place(x=330, y=150)

def finalFunction():
	sentence = str(InputText.get())
	print(sentence)
	sentence = lemmatize_sentence(sentence)
	result = sentiment_scores(sentence)
	print(result)
	showEmoji(result)

def CreateSuggestion():
	print("Called Suggestion")
	print(str(url.get()))
	if not str(url.get())=="nothing":
		webbrowser.open(str(url.get()),new=1)
	else:
		print("Please Enter Message and press Button to check your Emotion.")

lemmatizer = WordNetLemmatizer()

master = tk.Tk()
master.geometry("800x500")

url = StringVar()
url.set("nothing")

InputTextLabel = Label(master, text = "Enter Message: ", bg="black", fg="white").place(x = 30,y = 25)  
InputText = Entry(master, width="70")
InputText.place(x = 150, y = 25)  
checkEmotion = Button(master, text = "Check Emotion",activebackground = "black", activeforeground = "white", bg="white", command= finalFunction).place(x = 330, y = 65)

panel = tk.Label(master)
SadImagePath = "/home/nagadiapreet/Desktop/SDP_Final/Screenshots/sad.png"
SadEmotionImage = Image.open("sad.png")
SadEmotionImage = SadEmotionImage.resize((150, 150), Image.BILINEAR)
SadEmotionImage = ImageTk.PhotoImage(SadEmotionImage)

HappyImagePath = "/home/nagadiapreet/Desktop/SDP_Final/Screenshots/happy.png"
HappyEmotionImage = Image.open("happy.png")
HappyEmotionImage = HappyEmotionImage.resize((150, 150), Image.BILINEAR)
HappyEmotionImage = ImageTk.PhotoImage(HappyEmotionImage)

NeutralImagePath = "/home/nagadiapreet/Desktop/SDP_Final/Screenshots/neutral.png"
NeutralEmotionImage = Image.open("neutral.png")
NeutralEmotionImage = NeutralEmotionImage.resize((150, 150), Image.BILINEAR)
NeutralEmotionImage = ImageTk.PhotoImage(NeutralEmotionImage)

Suggestion = Button(master, text="Our Suggestion", activebackground = "black", activeforeground = "white", command= CreateSuggestion, bg="white", fg="black")
Suggestion.place(x=325, y=350)

neutralLinks = ['https://www.youtube.com/watch?v=1SZ92qTNWCk', 'https://www.youtube.com/watch?v=hE6I9apUvrk','https://www.youtube.com/watch?v=HHgQO_rKLHU','https://www.youtube.com/watch?v=l6B1sspQU1o','https://www.youtube.com/watch?v=gpuqFNGxab8']
happyLinks = ['https://www.youtube.com/watch?v=E_g263MKjhY','https://www.youtube.com/watch?v=1dpX1j_OPyg','https://www.youtube.com/watch?v=eszAgtg-Z9U','https://www.youtube.com/watch?v=v5R75oeyQFY','https://www.youtube.com/watch?v=Y5Wt-FBLGWk']
sadLinks = ['https://www.youtube.com/watch?v=eAK14VoY7C0','https://www.youtube.com/watch?v=rGpwRlCOLbY','https://www.youtube.com/watch?v=PgluKyEwuoE','https://www.youtube.com/watch?v=HHgQO_rKLHU','https://www.youtube.com/watch?v=gORVxUPTez0']

master.title("AI Therapist")
master.configure(background='black')

master.mainloop()