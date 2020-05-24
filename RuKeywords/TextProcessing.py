import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import re

import rutokenizer #https://github.com/Koziev/
import rupostagger #Anaconda Promt as Admin
#cd C:\Users\Виктория\Downloads\ruword2tags-master\ruword2tags-master 
###pip install -e .
import rulemma


class Preprocessor:
	
	def __init__(self, stopwordsList=None, lang='russian', *args, **kwargs):
		nltk.download("stopwords")
		#nltk.download("punkt")
		self.mystem = Mystem() 
		self.useLemmas = False

		if lang == 'russian': 
			self.lemmatizer = rulemma.Lemmatizer()
			self.lemmatizer.load()

			self.tokenizer = rutokenizer.Tokenizer()
			self.tokenizer.load()

			self.tagger = rupostagger.RuPosTagger()
			self.tagger.load()
		else:
			self.lemmatizer = WordNetLemmatizer()
		
		alphabet = []
		self.language = lang

		self.tag_dict = {"J": wordnet.ADJ,
					"N": wordnet.NOUN,
					"V": wordnet.VERB,
					"R": wordnet.ADV}

		if lang == 'russian':
			self.stopwords = stopwords.words("russian")
			alphabet = "АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя"
		else:
			self.stopwords = stopwords.words('english')
			alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
		self.stopwords.extend(list(alphabet))

		if not stopwordsList is None:
			self.stopwords.extend(stopwordsList)

	def getStopwords(self):
		return self.stopwords

	#Preprocess function
	def preprocess_text(self, text, removeStopWords=True, useLemmas=True, applyLemmasToText=True):
		
		'''
		if useLemmas == True:
			tokens = self.mystem.lemmatize(text.lower())

			if (removeStopWords == True):
				tokens = [token for token in tokens if token not in self.stopwords\
						  and token != " " \
						  and token.strip() not in punctuation]
			else:
				tokens = [token for token in tokens if token != " " \
						  and token.strip() not in punctuation]
		else:
			tokens = nltk.word_tokenize(text.lower())
			tokens= [word for word in tokens if word.isalnum()]
			if (removeStopWords == True):
				tokens = [token for token in tokens if self.mystem.lemmatize(token)[0] not in self.stopwords\
						  and token != " " \
						  and token.strip() not in punctuation]
			else:
				tokens = [token for token in tokens if token != " " \
						  and token.strip() not in punctuation]
			
		text = " ".join(tokens) 
		'''
		newText = None

		self.useLemmas = useLemmas
		if useLemmas == True and applyLemmasToText == True:
			newText = self.get_normal_form_text(text)
			#stopwordsList = self.get_normal_form_list(self.stopwords)
			#self.stopwords.extend(stopwordsList)
		else:
			newText = text

		if (removeStopWords == True):
			newText = self.removeStopwords(newText)
		
		return newText


	def get_wordnet_pos(self, word):
		"""Map POS tag to first character lemmatize() accepts"""
		if self.language != 'russian':
			tag = nltk.pos_tag([word])[0][1][0].upper()
			return self.tag_dict.get(tag, wordnet.NOUN)
		else:
			return None

	def get_normal_form(self, word):
		if self.language != 'russian':
			return self.lemmatizer.lemmatize(word, self.get_wordnet_pos(word))
		else:
			return None

	def get_normal_form_text(self, text):
		newText = ''
		if self.language != 'russian':
			#newText = ' '.join([self.lemmatizer.lemmatize(w, self.get_wordnet_pos(w))
			#for w in nltk.word_tokenize(text)])
			#maybe it will fasten the code
			newText = ' '.join([self.lemmatizer.lemmatize(w, self.tag_dict.get(nltk.pos_tag([w])[0][1][0].upper(), wordnet.NOUN)) for w in nltk.word_tokenize(text)])
			newText = re.sub(r'\s([?,.!"](?:\s|$))', r'\1', newText)
		else:
			tokens = self.tokenizer.tokenize(text)
			tags = self.tagger.tag(tokens)
			lemmas = self.lemmatizer.lemmatize(tags)
			newText = ' '.join([lemma[2] for lemma in lemmas]) #когда понадобиться индикатор istitle() смотреть на lemma[0]
			newText = re.sub(r'\s([?,.!"](?:\s|$))', r'\1', newText) 
		return newText

	def get_normal_form_list(self, lst):
		newList = []
		if self.language != 'russian':
			newList = [self.lemmatizer.lemmatize(w, self.tag_dict.get(nltk.pos_tag([w])[0][1][0].upper(), wordnet.NOUN)) for w in lst]
		else:
			tokens = self.tokenizer.tokenize(' '.join(lst))
			tags = self.tagger.tag(tokens)
			lemmas = self.lemmatizer.lemmatize(tags)
			newList = [lemma[2] for lemma in lemmas]
		return newList

	def removeStopwords(self, text):
		text_tokens = word_tokenize(text)
		tokens_without_sw = [word for word in text_tokens if not word in self.stopwords]
		filtered_text = (" ").join(tokens_without_sw)
		filtered_text = re.sub(r'\s([?,.!"](?:\s|$))', r'\1', filtered_text)
		return filtered_text