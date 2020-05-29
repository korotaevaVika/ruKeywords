from math import radians
import numpy as np     # installed with matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import statistics

import itertools

import Rake as rakeExtractor
from TfIdf import TfIdfBlobExtractor
from TextRank import TextRankExtractor
from test_ru_lemma import test_ru_lemma

from FileReader import Reader
from TextProcessing import Preprocessor
from tqdm import tqdm
from Metrics import evaluate_keywords
from tabulate import tabulate

import nltk
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense

def autolabel(ax, bar_plot, bar_label):
    for idx,rect in enumerate(bar_plot):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                bar_label[idx],
                ha='center', va='bottom', rotation=0)

def loadArticlesInEnglish(filenameTexts='texts.csv', filenameKeywords='keywords.csv', filenameMatches='text_keywords (1).csv'):
	texts = pd.read_csv(filenameTexts)
	#texts.head()
	keywords = pd.read_csv(filenameKeywords)
	keywords.columns = ['keyword_id', 'keyword']

	text_keywords = pd.read_csv(filenameMatches)
	text_keywords.columns = ['text_id', 'keyword_ids']

	texts_with_keywords = texts.copy()
	texts_with_keywords['Keywords'] = ''
	for index, row in tqdm(texts_with_keywords.iterrows(), total=texts_with_keywords.shape[0]):
		text_id = row.text_id
		#s = text_keywords[text_keywords.text_id ==
		#text_id].keyword_ids.str.split(expand=True).at[text_id, 0]
		#print(s)
		key_ids = text_keywords[text_keywords.text_id == text_id].keyword_ids.str.split(expand=True).at[text_id, 0].split(',')#get_value(text_id, 0).split(',')
		texts_with_keywords.at[index, 'Keywords'] = ','.join(keywords[(keywords.keyword_id.isin(key_ids))]['keyword'].values)

	texts_with_keywords = texts_with_keywords.rename(columns={"text": 'Source Text'})

	return texts_with_keywords

def main():
	#lang = 'english'
	lang = 'russian'
	partMatchesCounted = False #if true then custom, else rough precision recall
	num = 8

	if lang == 'russian':
		reader = Reader('elsevier journal.pdf') #, language='russian')
		reader.loadFile()
		#pages = reader.readPages();

		##pages = reader.readPages(0, None, 'НАПРАВЛЕНИЕ 1', 'НАПРАВЛЕНИЕ 2',
		##caseSensitiveSearch=True)
		##pages = reader.readPages(0, None, 'НАПРАВЛЕНИЕ 1', 'НАПРАВЛЕНИЕ 2',
		##caseSensitiveSearch=False)
		##pages = reader.readPages(start=2, end=10, startString=None, endString=None,
		##debug=True)
		##pages = reader.readPages(start=4, end=4, startString=None, endString=None,
		##debug=True)
		#print(len(pages))

		words, lines, articles = reader.parseDocPages()#2160, 2190)#0, 50)
		print(len(articles))

		df = reader.getArticlesDataframe()
	
	else:
		df = loadArticlesInEnglish()

	df['text'] = ''

	processor = Preprocessor(stopwordsList=None, lang=lang)
	
	if (lang == 'russian'):
		df['text'] = df['Source Text']
		#for index, row in tqdm(df.iterrows(), total=df.shape[0]):
		#	df.at[index, 'text'] = processor.preprocess_text(row['Source Text'],
		#	useLemmas = False)
	else:
		df['text'] = df['Source Text']
		#for index, row in tqdm(df.iterrows(), total=df.shape[0]):
			#df.at[index, 'text'] = processor.preprocess_text(row['Source Text'],
			#removeStopWords=False, useLemmas=False)

	print(df.head())
	rakeExtractor.extractKeywords(df, num=num, metricsCount=True, partMatchesCounted=partMatchesCounted)

	print(df.head())
	#print(tabulate(df, headers='keys', tablefmt='psql'))
	kw = df['Keywords'].values
	#print(kw)
	#print(df.Keywords)
	
	tfidfBlobExtractor = TfIdfBlobExtractor(processor.getStopwords())
	tfidfBlobExtractor.extractKeywords(df, num=num, metricsCount=True, partMatchesCounted=partMatchesCounted)
	
	#print(df.head())

	textRankExtractor = TextRankExtractor(processor.getStopwords(), language=lang)
	textRankExtractor.extractKeywords(df, num=num, metricsCount=True, partMatchesCounted=partMatchesCounted)#False)
	
	#print(df.head())

	#print(tabulate(df.head(), headers='keys', tablefmt='psql'))

	x = []
	y = {'rake': {'precision':[], 'recall':[], 'f1':[]},
	 'textrank': {'precision':[], 'recall':[], 'f1':[]}, 
	 'tfidf': {'precision':[], 'recall':[], 'f1':[]}}

	for index, row in tqdm(df.iterrows(), total=df.shape[0]):
		x.append(index)

		values = row['textrank_metrics'].split(',')
		y['textrank']['precision'].append(values[0])
		y['textrank']['recall'].append(values[1])
		y['textrank']['f1'].append(values[2])
		
		values = row['tfidf_blob_metrics'].split(',')
		y['tfidf']['precision'].append(values[0])
		y['tfidf']['recall'].append(values[1])
		y['tfidf']['f1'].append(values[2])

		values = row['rake_metrics'].split(',')
		y['rake']['precision'].append(values[0])
		y['rake']['recall'].append(values[1])
		y['rake']['f1'].append(values[2])
	#plt.plot(x, y['textrank']['precision'], 'g^', x, y['textrank']['recall'],
	#'g-')
	
	#fig, ax = plt.subplots()

	#bar_values = [statistics.mean(list(map(float, y['textrank']['precision']))),
	#			statistics.mean(list(map(float, y['rake']['precision']))),
	#			statistics.mean(list(map(float, y['tfidf']['precision'])))]
	#bar_label = bar_values

	#bar_plot = plt.bar(['textrank', 'rake', 'tf-idf'], bar_values)

	#autolabel(ax, bar_plot, bar_label)
	#plt.ylim(0,max(bar_label) * 1.5)
	#plt.title('Quality metrics for ' + lang + ' language')
	#plt.savefig("add_text_bar_matplotlib_01.png", bbox_inches='tight')
	#plt.show()

	metrics = ['precision', 'recall', 'f1']
	for i in range(len(metrics)):
		fig, ax = plt.subplots()
		bar_values = [statistics.mean(list(map(float, y['textrank'][metrics[i]]))),
				statistics.mean(list(map(float, y['rake'][metrics[i]]))),
				statistics.mean(list(map(float, y['tfidf'][metrics[i]])))]
		bar_label = bar_values
		bar_plot = plt.bar(['textrank', 'rake', 'tf-idf'], bar_values)
		autolabel(ax, bar_plot, bar_label)
		plt.ylim(0,max(max(bar_label) * 1.5, 0.01))
		title = 'Metric ' + str(metrics[i] + ' for ' + str(num) + ' found keywords based on data in ' + lang + ' (partMatchesCounted = ' + str(partMatchesCounted) + ')')
		plt.title(title)#'Quality metrics for ' + lang + ' language')
		plt.savefig(title + ".png", bbox_inches='tight')
	
	plt.show()


#precision, recall, f1 = evaluate_keywords(['ключевая фраза', 'слово'], [
#'фраза', 'ключевое слово'], True )
#print(precision, recall, f1)
def draw():
	x = np.linspace(0, 2 * np.pi, 400)
	y = np.sin(x ** 2)

	# Create just a figure and only one subplot
	fig, ax = plt.subplots()
	ax.plot(x, y)
	ax.set_title('Simple plot')

	# Create two subplots and unpack the output array immediately
	f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
	ax1.plot(x, y)
	ax1.set_title('Sharing Y axis')
	ax2.scatter(x, y)

	# Create four polar axes and access them through the returned array
	fig, axs = plt.subplots(2, 2, subplot_kw=dict(polar=True))
	axs[0, 0].plot(x, y)
	axs[1, 1].scatter(x, y)

	# Share a X axis with each column of subplots
	plt.subplots(2, 2, sharex='col')

	for i in range(3):
		fig, ax = plt.subplots()
		bar_values = [i * 2, i * 1, i * 3]
		bar_label = bar_values
		bar_plot = plt.bar(['textrank', 'rake', 'tf-idf'], bar_values)
		autolabel(ax, bar_plot, bar_label)
		plt.ylim(0,max(max(bar_label) * 1.5, 0.01))
		title = 'title ' + str(i)
		plt.title(title)#'Quality metrics for ' + lang + ' language')
		plt.savefig(title + ".png", bbox_inches='tight')

	#plt.show()
	## Share a Y axis with each row of subplots
	#plt.subplots(2, 2, sharey='row')

	## Share both X and Y axes with all subplots
	#plt.subplots(2, 2, sharex='all', sharey='all')

	## Note that this is the same as
	#plt.subplots(2, 2, sharex=True, sharey=True)

	# Create figure number 10 with a single subplot
	# and clears it if it already exists.
	#fig, ax = plt.subplots(num=10, clear=True)

	plt.show()


def lemmaEng():
	lang = 'english'
	#lang = 'russian'
	partMatchesCounted = False #if true then custom, else rough precision recall
	num = 8

	if lang == 'russian':
		reader = Reader('elsevier journal.pdf') #, language='russian')
		reader.loadFile()
		
		words, lines, articles = reader.parseDocPages()#2160, 2190)#0, 50)
		print(len(articles))

		df = reader.getArticlesDataframe()
	
	else:
		df = loadArticlesInEnglish()

	df['text'] = ''

	processor = Preprocessor(stopwordsList=None, lang=lang)
	
	if (lang == 'russian'):
		df['text'] = df['Source Text']
		#for index, row in tqdm(df.iterrows(), total=df.shape[0]):
		#	df.at[index, 'text'] = processor.preprocess_text(row['Source Text'],
		#	useLemmas = False)
	else:
		df['text'] = df['Source Text']
		subset = df.iloc[0:1 , :]
		
		for index, row in tqdm(subset.iterrows(), total=subset.shape[0]):
			text_lemma_sw = processor.preprocess_text(row['Source Text'], removeStopWords=True, useLemmas=True)
			text_lemma = processor.preprocess_text(row['Source Text'], removeStopWords=False, useLemmas=True)
			subset.at[index, 'text_lemma_sw'] = text_lemma_sw
			subset.at[index, 'text_lemma'] = text_lemma #processor.preprocess_text(row['Source Text'], removeStopWords=False,
                                               #useLemmas=True)
			source = row['Source Text']
			
def lemmaEngWithQuality():
	lang = 'english'

	if lang == 'russian':
		reader = Reader('elsevier journal.pdf') #, language='russian')
		reader.loadFile()
		
		words, lines, articles = reader.parseDocPages()#2160, 2190)#0, 50)
		print(len(articles))

		df = reader.getArticlesDataframe()
	
	else:
		df_source = loadArticlesInEnglish()


	
	#partMatchesCounted = False #if true then custom, else rough precision recall
	#num = 8;
	#removeStopWords = False


	conditions = list(itertools.product([True], #[False, True], #removeStopWords
		[False, True],	#partMatchesCounted #if true then custom, else rough precision recall
		[4, 8]	#num of keywords
	))

	df = None

	for condition in conditions:
		removeStopWords = condition[0] 
		partMatchesCounted = condition[1] #if true then custom, else rough precision recall
		num = condition[2] 
		
		print('condition: ', condition)

		if df is None:
			print('Read DF')
			df = df_source.copy()
			df['text'] = ''
			processor = Preprocessor(stopwordsList=None, lang=lang)
			sw = processor.stopwords
			processor.stopwords = processor.get_normal_form_list(sw)

			if (lang == 'russian'):
				df['text'] = df['Source Text']
				#for index, row in tqdm(df.iterrows(), total=df.shape[0]):
				#	df.at[index, 'text'] = processor.preprocess_text(row['Source Text'],
				#	useLemmas = False)
			else:
				for index, row in tqdm(df.iterrows(), total=df.shape[0]):
					text = processor.preprocess_text(row['Source Text'], removeStopWords=removeStopWords, useLemmas=True)
					df.at[index, 'text'] = text

		rakeExtractor.extractKeywords(df, num=num, metricsCount=True, partMatchesCounted=partMatchesCounted)
	
		tfidfBlobExtractor = TfIdfBlobExtractor(processor.getStopwords())
		tfidfBlobExtractor.extractKeywords(df, num=num, metricsCount=True, partMatchesCounted=partMatchesCounted)
	
		textRankExtractor = TextRankExtractor(processor.getStopwords(), language=lang)
		textRankExtractor.extractKeywords(df, num=num, metricsCount=True, partMatchesCounted=partMatchesCounted)
				
		x = []
		y = {'rake': {'precision':[], 'recall':[], 'f1':[]},
		 'textrank': {'precision':[], 'recall':[], 'f1':[]}, 
		 'tfidf': {'precision':[], 'recall':[], 'f1':[]}}

		for index, row in tqdm(df.iterrows(), total=df.shape[0]):
			x.append(index)

			values = row['textrank_metrics'].split(',')
			y['textrank']['precision'].append(values[0])
			y['textrank']['recall'].append(values[1])
			y['textrank']['f1'].append(values[2])
		
			values = row['tfidf_blob_metrics'].split(',')
			y['tfidf']['precision'].append(values[0])
			y['tfidf']['recall'].append(values[1])
			y['tfidf']['f1'].append(values[2])

			values = row['rake_metrics'].split(',')
			y['rake']['precision'].append(values[0])
			y['rake']['recall'].append(values[1])
			y['rake']['f1'].append(values[2])

		metrics = ['precision', 'recall', 'f1']
		for i in range(len(metrics)):
			fig, ax = plt.subplots()
			bar_values = [statistics.mean(list(map(float, y['textrank'][metrics[i]]))),
					statistics.mean(list(map(float, y['rake'][metrics[i]]))),
					statistics.mean(list(map(float, y['tfidf'][metrics[i]])))]

			bar_label = [round(bv, 2) for bv in bar_values] #add round
			
			bar_plot = plt.bar(['textrank', 'rake', 'tf-idf'], bar_values)
			autolabel(ax, bar_plot, bar_label)
			plt.ylim(0,max(max(bar_label) * 1.5, 0.01))
			title = ('Metric ' + str(metrics[i] + ' for ' + str(num) + ' found keywords based on lemmatizied data in ' + lang + ' (partMatchesCounted = ' + str(partMatchesCounted) + ')' + ' (removeStopWords = ' + str(removeStopWords) + ')'))
			plt.title(title)#'Quality metrics for ' + lang + ' language')
			plt.savefig(title + ".png", bbox_inches='tight')
	
			#if want to show only once upon a program
	plt.show()

def test_ru():
	#obj = test_ru_lemma()
	#obj.test()
	preprocessor = Preprocessor(None, 'russian')
	lst = preprocessor.getStopwords()
	print(lst)
	lst1 = preprocessor.get_normal_form_list(lst)
	print(lst1)

def lemmaRuWithQuality():
	lang = 'russian'

	if lang == 'russian':
		reader = Reader('elsevier journal.pdf') #, language='russian')
		reader.loadFile()
		
		words, lines, articles = reader.parseDocPages()#0, 50)#()#2160, 2190)#0, 50)
		print(len(articles))

		df_source = reader.getArticlesDataframe()
	
	else:
		df_source = loadArticlesInEnglish()

	#partMatchesCounted = False #if true then custom, else rough precision recall
	#num = 8;
	#removeStopWords = False


	conditions = list(itertools.product([False, True], #[False, True], #removeStopWords
		[False, True],	#partMatchesCounted #if true then custom, else rough precision recall
		[4, 8]	#num of keywords
	))

	df = None

	condition_i = 0

	for condition in conditions:
		removeStopWords = condition[0] 
		partMatchesCounted = condition[1] #if true then custom, else rough precision recall
		num = condition[2] 
		
		print('condition: ', condition)

		if df is None:
			print('Read DF')
			df = df_source.copy()
			df['text'] = ''
			processor = Preprocessor(stopwordsList=None, lang=lang)
			sw = processor.stopwords
			#processor.stopwords = processor.get_normal_form_list(sw)
			processor.stopwords.extend(processor.get_normal_form_list(sw))

			if (lang == 'russian'):
				#df['text'] = df['Source Text']
				for index, row in tqdm(df.iterrows(), total=df.shape[0]):
					df.at[index, 'text'] = processor.preprocess_text(row['Source Text'], 
													removeStopWords=removeStopWords, 
													useLemmas = True, 
													applyLemmasToText=True)
			else:
				for index, row in tqdm(df.iterrows(), total=df.shape[0]):
					text = processor.preprocess_text(row['Source Text'], removeStopWords=removeStopWords, useLemmas=True)
					df.at[index, 'text'] = text

		elif condition_i == 4:
			for index, row in tqdm(df.iterrows(), total=df.shape[0]):
				text = processor.preprocess_text(row['text'], removeStopWords=True, useLemmas=True, applyLemmasToText=False)
				df.at[index, 'text'] = text
		
		condition_i = condition_i + 1

		rakeExtractor.extractKeywords(df, num=num, metricsCount=True, partMatchesCounted=partMatchesCounted, textprocessor=processor)
	
		tfidfBlobExtractor = TfIdfBlobExtractor(processor.getStopwords(), textprocessor=processor)
		tfidfBlobExtractor.extractKeywords(df, num=num, metricsCount=True, partMatchesCounted=partMatchesCounted)
	
		textRankExtractor = TextRankExtractor(processor.getStopwords(), language=lang, textprocessor=processor)
		textRankExtractor.extractKeywords(df, num=num, metricsCount=True, partMatchesCounted=partMatchesCounted)
				
		x = []
		y = {'rake': {'precision':[], 'recall':[], 'f1':[]},
		 'textrank': {'precision':[], 'recall':[], 'f1':[]}, 
		 'tfidf': {'precision':[], 'recall':[], 'f1':[]}}

		for index, row in tqdm(df.iterrows(), total=df.shape[0]):
			x.append(index)

			values = row['textrank_metrics'].split(',')
			y['textrank']['precision'].append(values[0])
			y['textrank']['recall'].append(values[1])
			y['textrank']['f1'].append(values[2])
		
			values = row['tfidf_blob_metrics'].split(',')
			y['tfidf']['precision'].append(values[0])
			y['tfidf']['recall'].append(values[1])
			y['tfidf']['f1'].append(values[2])

			values = row['rake_metrics'].split(',')
			y['rake']['precision'].append(values[0])
			y['rake']['recall'].append(values[1])
			y['rake']['f1'].append(values[2])

		metrics = ['precision', 'recall', 'f1']
		for i in range(len(metrics)):
			fig, ax = plt.subplots()
			bar_values = [statistics.mean(list(map(float, y['textrank'][metrics[i]]))),
					statistics.mean(list(map(float, y['rake'][metrics[i]]))),
					statistics.mean(list(map(float, y['tfidf'][metrics[i]])))]

			bar_label = [round(bv, 2) for bv in bar_values] #add round
			
			bar_plot = plt.bar(['textrank', 'rake', 'tf-idf'], bar_values)
			autolabel(ax, bar_plot, bar_label)
			plt.ylim(0,max(max(bar_label) * 1.5, 0.01))
			title = ('Metric ' + str(metrics[i] + ' for ' + str(num) + ' found keywords based on lemmatizied data in ' + lang + ' (partMatchesCounted = ' + str(partMatchesCounted) + ')' + ' (removeStopWords = ' + str(removeStopWords) + ')'))
			plt.title(title)#'Quality metrics for ' + lang + ' language')
			plt.savefig(title + ".png", bbox_inches='tight')
	
			#if want to show only once upon a program
	plt.show()

#main()
#draw()
#lemmaEng()
#lemmaEngWithQuality()
#test_ru()
#lemmaRuWithQuality()
def ru_words_test():
	lang = 'russian'

	if lang == 'russian':
		reader = Reader('elsevier journal.pdf', generateWordList = True) #, language='russian')
		reader.loadFile()
		
		words, lines, articles = reader.parseDocPages(19, 21)#(14, 17)#0, 50)#()#2160, 2190)#0, 50)
		
		#df_source = reader.getArticlesDataframe()

		#print(words)
		#print(' '.join([x['text'] for x in words]))

		#txt = ' '.join([x['text'] for x in words if x['article_num'] == 0])
		#words_0 = [x for x in words if x['article_num'] == 0]

		#for i, w in enumerate(words_0):
		#	if (i > 50 and i < 80):
		#		print(w['text'], ' ', w['sentence_count'], ' ', w['word_pos_in_sentence'])
		
		for i in range(11):
			print(i)
			if i != 6:
				print(' '.join([x['text'] for x in words if x['block_count'] == i]))
			else:
				for j in range(15):
					print(j, ' sentence')
					print(' '.join([x['text'] for x in words if x['block_count'] == i and x['sentence_count'] == j]))
			

		#print([x['text'] for x in words_0 if x['is_keyword'] > 0])

		#print([x['text'] for x in words_0 if x['textrank_score'] != 0])

		#for i, w in enumerate(words_0):
		#	if (i > 50 and i < 80):
		#		print(w['text'], ' ', w['sentence_count'], ' ', w['word_pos_in_sentence'])

		#for j in range(23):
		#	print(j, ' block')
		#	for i in range(10):
		#		print(i, ' sentence')
		#		print(' '.join([x['text'] for x in words if x['article_num'] == 0 and
		#		x['sentence_count'] == i
		#			and x['block_count'] == j]))
		#	print('-----')
		print("--=-")
		#print(txt)
		#print("test")
	else:
		df_source = loadArticlesInEnglish()

import numpy as np
import pandas as pd

#ru_words_test();
def SimpleNN():
	lang = 'russian'

	nltk.download("stopwords")
	lst_stopwords = stopwords.words("russian")
	alphabet = "АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя"
	alphabet += "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
	lst_stopwords.extend(list(alphabet))
	#print(lst_stopwords)

	if lang == 'russian':
		reader = Reader('elsevier journal.pdf',
				  generateWordList = True, 
				  additional_stopwords=lst_stopwords) #, language='russian')
		reader.loadFile()
		
		words, lines, articles = reader.parseDocPages(19, 20) #1)		
		#print([x['text'] for x in words if x['is_stopword'] != 0])
		#print([x['text'] for x in words if x['frequency'] > 2])

		df = pd.DataFrame.from_dict(words)

		cat_features = ['size', 'flags', 'font', #'otherSigns', 
		'morph_pos', 'morph_animacy', 'morph_aspect', 'morph_case',
		'morph_gender', 'morph_involvement', 'morph_mood',
		'morph_number', 'morph_person', 'morph_tense',
		'morph_transitivity', 'morph_voice']

		all_columns = list(df.columns)
		df = pd.concat([pd.get_dummies(df[col], prefix=col)
							if col in cat_features and col != 'otherSigns' else df[col] for col in all_columns]
							, axis=1)

		values = [',', '.', '\)', '\(', '\[', '\]']

		for value in values:
			df[str('otherSigns' + '_' + value)] = np.where(df['otherSigns'].str.contains(value), "1", "0") 
		df = df.drop(['otherSigns'], axis=1)

		print('all columns')
		for col in all_columns:
			print(col)
		print()

		print('new columns')
		for col in df.columns:
			print(col)

		df.head()

		#for feature in features:
		#	x = np.array([w[feature] for w in words])
		#	unique, counts = np.unique(x, return_counts=True)

		#	if (feature == 'morph_pos'):
		#		possible_options = ['NOUN', 'ADJF','ADJS','COMP','VERB',
		#				'INFN','PRTF','PRTS','GRND','NUMR','ADVB','NPRO',
		#				'PRED','PREP','CONJ','PRCL','INTJ']

		#		data_to_add = []
		#		for _x in x:
		#			data_to_add.append([1 if v == _x else 0 for v in possible_options ])

		#		all_data = append_fields(all_data,
		#					 possible_options, 
		#					data_to_add,
		#					usemask=False)

		#	print(feature)
		#	print(len(counts))
		#	print("--=--")

		#x = words[0]

		#print(x['size'], 
		#x['flags'],
		#x['font'], 
		#x['color'],
		#x['otherSigns'], 
		#x['istitle'], 
		#x['isupper'], 
		#x['is_stopword'],
		#x['text'],
		#x['textrank_score'],
		#x['frequency'],
		#x['morph_score'], 
		#x['morph_pos'],
		#x['morph_animacy'], 
		#x['morph_aspect'],
		#x['morph_case'], 
		#x['morph_gender'],
		#x['morph_involvement'], 
		#x['morph_mood'],
		#x['morph_number'], 
		#x['morph_person'],
		#x['morph_tense'],
		#x['morph_transitivity'], 
		#x['morph_voice'],
		#x['morph_isnormal'], 
		#x['morph_normalform'], 
		#x['morph_lexeme'],
		#x['word_pos_in_sentence'])

		#X = []
		#for x in words:
		#	X.append([x['size'], x['flags'],x['font'], x['color'], x['otherSigns'], x['istitle'], 
		#			  x['isupper'], x['is_stopword'],
		#x['text'], x['textrank_score'],
		#x['frequency'],
		#x['morph_score'], x['morph_pos'],
		#x['morph_animacy'], x['morph_aspect'],
		#x['morph_case'], x['morph_gender'],
		#x['morph_involvement'], x['morph_mood'],
		#x['morph_number'], x['morph_person'],
		#x['morph_tense'], x['morph_transitivity'], 
		#x['morph_voice'], x['morph_isnormal'], 
		#x['morph_normalform'], x['morph_lexeme'],
		#x['word_pos_in_sentence']])
		#print(X[0:2])
		
SimpleNN()