import fitz #to read doc
import unicodedata # to remove ugly hex symbols from raw text data in doc
import re
import pandas as pd
from tqdm import tqdm


from nltk.corpus import wordnet
import rutokenizer #https://github.com/Koziev/
import rupostagger #Anaconda Promt as Admin
#cd C:\Users\Виктория\Downloads\ruword2tags-master\ruword2tags-master
###pip install -e .
import rulemma

import pymorphy2

from django.utils.encoding import smart_text
from summa import keywords as textrank

class Reader:

	def __init__(self, fileName, generateWordList=False, additional_stopwords=None): 	
		self.fileName = fileName
		self.doc = None
		self.pageObjects = []
		self.pages = []
		self.words = []
		self.lines = []
		self.articles = []
		self.stopwords = additional_stopwords

		self.generateWordList = generateWordList
		#self.lemmatizer = rulemma.Lemmatizer()
		#self.lemmatizer.load()
		
		#self.tokenizer = rutokenizer.Tokenizer()
		#self.tokenizer.load()
		
		#self.tagger = rupostagger.RuPosTagger()
		#self.tagger.load()

		self.morph = pymorphy2.MorphAnalyzer()

	def loadFile(self):
		self.doc = fitz.open(self.fileName)
		print("number of pages: %i" % self.doc.pageCount)
		print(self.doc.metadata)

	def readPages(self,
				   start=0, 
				   end=None, 
				   startString=None, 
				   endString=None, 
				   caseSensitiveSearch=True, 
				   debug=True):
		count = start
		text = ""
		if (startString is not None):
			startStringPage = -1
		else:
			startStringPage = None

		if (endString is not None):
			endStringPage = -1
		else:
			endStringPage = None

		if caseSensitiveSearch == False:
			startString = startString.lower()
			endString = endString.lower()

		if (end is None) or (end > self.doc.pageCount):
			end = self.doc.pageCount - 1

		while count <= end:        
			pageObject = self.doc.loadPage(count)
			pageText = pageObject.getText("text")

			if (startStringPage is not None and startStringPage < 0):
				if ((caseSensitiveSearch == True and pageText.find(startString) >= 0) or (caseSensitiveSearch == False and pageText.lower().find(startString) >= 0)):
					startStringPage = count
					if debug == True:
						print('startStringPage = ' + str(startStringPage))

			if (endStringPage is not None and endStringPage < 0):
				if ((caseSensitiveSearch == True and pageText.find(endString) >= 0) or (caseSensitiveSearch == False and pageText.lower().find(endString) >= 0)):
					endStringPage = count
					end = count

			if not (startStringPage is not None and startStringPage < 0):
				self.pageObjects.append(pageObject)
				self.pages.append(pageText)
				
			count +=1
		if debug == True:
			print("Last page printed = " + str(end))
		return self.pages

	def getChapterList(self):
		#начала разделов...
		return self.doc.getToC()

	"""
	Return words, lines, articles lists
	"""
	def parseDocPages(self, pageStart=1, pageEnd=None, includeRemarks=False):
		self.words = []
		self.lines = []
		self.articles = []
		#regex = re.compile('[^а-яА-Я]')

		if (pageEnd is None):
			pageEnd = self.doc.pageCount 
		elif self.doc.pageCount < pageEnd:
			pageEnd = self.doc.pageCount
		pageNum = pageStart

		articleCount = -1
		isNewArticle = False
		author = ''
		university = ''
		mail = ''
		title = ''
		isKeywordsSpan = False
		keywords = ''
		normalized_keywords = None
		wordsCounter = {}

		#tag_dict = {"J": wordnet.ADJ,
		#			"N": wordnet.NOUN,
		#			"V": wordnet.VERB,
		#			"R": wordnet.ADV}
		block_count = 0

		while (pageNum <= pageEnd):
			page = self.doc.loadPage(pageNum - 1)
			dictionary = page.getText("dict")
			#block_count = 0
			goToNextPage = False
			
			for x in dictionary['blocks'][:]:
				authorsBlock = False
				isKeywordsSpan = False

				if ('image' in x):
					#do nothing if image block
					continue
				else:
					line_count = 0
					
					#TO DO To-do
					sentence_count = 0
					dot_found = False
					word_pos_in_sentence = 0
					block_changed = True

					if ((len(x['lines']) == 1) and (len(x['lines'][0]['spans']) == 1) and (x['lines'][0]['spans'][0]['text'].isdigit())):
						continue #do not write page number

					else:
						for line in x['lines']:
							span_count = 0
							processSpans = True

							if (block_count == 0 and line_count == 0 and 'Bold' in line['spans'][0]['font']):

								isNewArticle = True
								processSpans = False
								authorsBlock = True
								author = smart_text(line['spans'][0]['text'])
								university = ''
								mail = ''
								title = ''
								keywords = ''
								isKeywordsSpan = False
								normalized_keywords = None
								wordsCounter = {}

								self.UpdateTextrankScore(articleCount - 1)

							elif (block_count == 0 and authorsBlock == True and 'Bold' in line['spans'][0]['font']):

								author += smart_text(line['spans'][0]['text'])
								processSpans = False
								
							if (processSpans == True):

								for span in line['spans']:

									if (isNewArticle == True and 'Italic' in span['font'] and int(span['size']) == 10 and mail == ''):
										university += smart_text(span['text'])

									elif (isNewArticle == True and 'DJGIDG+SchoolBookC' in span['font'] and int(span['size']) == 8):
										mail += smart_text(span['text'])

									elif (isNewArticle == True and 'Bold' in span['font']):
										title += "".join(re.findall("[а-яА-Яa-zA-Z,ёЁ\s-]+", smart_text(span['text'])))#+

									elif (isNewArticle == True and 'Italic' in span['font'] and int(span['size']) == 8 and 'Ключевые слова' in smart_text(span['text']) and not title == '' and keywords == ''):
										isKeywordsSpan = True

									elif (isNewArticle == True and isKeywordsSpan == True):

										removeDotsWhenKeywordsStart = False
										#remove переносы по слогам
										if (len(keywords) > 0 and keywords[-1] == '-'):
											keywords = keywords[:-1]
										elif (keywords == '' and smart_text(span['text'])[0] == ':'):
											removeDotsWhenKeywordsStart = True

										keywords+= smart_text(span['text'])

										if (removeDotsWhenKeywordsStart == True):
											keywords = keywords[1:]

										if (len(smart_text(span['text']).strip()) > 0 and smart_text(span['text']).strip()[-1] == '.'):
											articleCount += 1
											isNewArticle = False
											keywords = keywords[:-1].strip() #remove last dot and all useless spaces
											author = author.strip()
											university = university.strip()
											mail = mail.strip()
											title = title.strip()
											normalized_keywords = [self.morph.parse(w)[0].normal_form.strip().split() \
												for w in keywords.split(',')]

											self.articles.append([articleCount, author, university, mail, title, keywords])
											author = ''
											university = ''
											mail = ''
											title = ''
											keywords = ''
											isKeywordsSpan = False
											
									elif (isNewArticle == False and (includeRemarks == False and int(span['size']) > 8)):   #not example remarks

										lineText = ''
										prevWord = None
										# удаление переноса по слогам в предыдущей строке
										if (span_count == 0 and len(self.lines) > 0 and self.lines[-1]['text'][-1] == '-' and smart_text(span['text'])[0].islower() == True):
											
											prevWord = self.words.pop()
											n = len(prevWord['text']) + 1 # +1 на дефис
											self.lines[-1]['text'] = self.lines[-1]['text'][:-n]
											prevWord['text'] = prevWord['text'][:-1]

											if (prevWord != None and len(prevWord['text'][0]) > 0 and not(prevWord['text'][0] == ' ' or prevWord['text'][0] == '')):
												lineText += ' '
											
											lineText += prevWord['text'] 

										if (('НАПРАВЛЕНИE' in smart_text(span['text']).upper()) or span['size'] == 13):
											#switch to the next page
											goToNextPage = True
											break

										lineText += smart_text(span['text'].replace(u'\xa0', u' '))
										
										self.lines.append({'size':span['size'], 
														  'flags': span['flags'],
														  'font': span['font'],
														  'color': span['color'],
														  'text': lineText,
														  'span_count': span_count,
														  'line_count': line_count,
														  'block_count': block_count,
														  'page_num': pageNum,
														  'article_num': articleCount
													 })

										
										for txt in lineText.split(' '):
											
											#запишем слово вместе с его слогами с предыдущей строки
											if not prevWord is None and not (txt in [" ", ""]):
												word = prevWord['text'] + txt
												otherSigns = prevWord['otherSigns']
												prevWord = None
											else:
												word = txt
												otherSigns = ""

											otherSigns += "".join(re.findall("[^а-яА-Яa-zA-Z,ёЁ-]", word))
											
											if (not otherSigns == ""): #(regex.match(word)):
												word = "".join(re.findall("[а-яА-Яa-zA-Z,ёЁ-]+", word))
											
											if (self.generateWordList == True and word == "" and '.' in otherSigns):
												self.words[-1]['otherSigns'] = otherSigns
												dot_found = True

											if (word == ""):
												prevWord = None
												continue

											if (self.generateWordList == False):
												self.words.append({'text': word})
												continue

											x = self.morph.parse(word)[0]
											
											if (dot_found == False and '.' in otherSigns and not '..' in otherSigns):
												dot_found = True

											else:
												if (dot_found and len(word) > 0 and word[0].isupper() and (self.words[-1] is not None and self.words[-1]['isupper'] == False)):
													
													dot_found = False
													word_pos_in_sentence = 0
													words_in_sentence = self.words[-1]['word_pos_in_sentence']
													
													if (sentence_count > 1):
														for i, w in enumerate(self.words):

															if (w['article_num'] == articleCount and w['block_count'] == block_count and w['sentence_count'] == sentence_count):
																w['word_pos_in_sentence'] /= words_in_sentence
																											
													sentence_count += 1
												elif dot_found and len(word) > 0 and word[0].islower():
													dot_found = False

											if not normalized_keywords is None and len(normalized_keywords) > 0:
												is_keyword = max([1 for k in normalized_keywords \
													if x.normal_form in k], default=0) # 1 -> 1 / len(k); max -> sum
											else:
												is_keyword = 0
											
											if not self.stopwords is None and len(self.stopwords) > 0:
												is_stopword = max([1 for k in self.stopwords \
													if x.normal_form == k], default=0) # 1 -> 1 / len(k); max -> sum
											else:
												is_stopword = 0
											
												
											if block_changed:
												if len(self.words) > 0 and not '.' in self.words[-1]['otherSigns']:
													sentence_count = self.words[-1]['sentence_count']
													block_count -= 1
												block_changed = False
											
											word_normal_form = x.normal_form
											if x.tag.POS in ['ADJF', 'ADJS', 'PRTF', 'PRTS']:
												word_normal_form = x.inflect({'sing', 'masc', 'nomn'}).word

											wordsCounter[word_normal_form] = wordsCounter.get(word_normal_form, 0) + 1

											self.words.append({'size':span['size'], 
														  'flags': span['flags'],
														  'font': span['font'],
														  'color': span['color'],
														  'otherSigns': otherSigns,
														  'istitle': word[0:2].istitle(),
														  'isupper': word.isupper(),
														  'is_keyword': is_keyword,
														  'is_stopword': is_stopword,
														  'text': word, #unicodedata.normalize('NFKC', txt),
														  'textrank_score': 0,
														  'frequency': wordsCounter.get(word_normal_form),

														  'morph_score' : x.score,
														  'morph_pos' : x.tag.POS, # Part of Speech, часть речи,
														  'morph_animacy' : x.tag.animacy, # одушевленность
														  'morph_aspect' : x.tag.aspect, # вид: совершенный или несовершенный
														  'morph_case' : x.tag.case, # падеж
														  'morph_gender' : x.tag.gender, # род (мужской, женский, средний)
														  'morph_involvement' : x.tag.involvement, # включенность говорящего в действие
														  'morph_mood' : x.tag.mood, # наклонение (повелительное, изъявительное)
														  'morph_number' : x.tag.number, # число (единственное, множественное)
														  'morph_person' : x.tag.person, # лицо (1, 2, 3)
														  'morph_tense' : x.tag.tense, # время (настоящее, прошедшее, будущее)
														  'morph_transitivity' : x.tag.transitivity, # переходность (переходный, непереходный)
														  'morph_voice' : x.tag.voice, # залог (действительный, страдательный)
														  'morph_isnormal' : x.normal_form == x.word, # слово в неопределенной форме : да / нет
														  'morph_normalform' : word_normal_form, # неопределенная форма
														  'morph_lexeme' : x.lexeme[0][0], # лексема
														  
														  'word_pos_in_sentence': word_pos_in_sentence,
														  'span_count': span_count,
														  'line_count': line_count,
														  'block_count': block_count,
														  'sentence_count': sentence_count,
														  'page_num': pageNum,
														  'article_num': articleCount
													 })

											word_pos_in_sentence += 1
										span_count += 1
							line_count += 1
							
							if goToNextPage == True:
								break

						if goToNextPage == True:
							break

						block_count += 1
						sentence_count = 0
						dot_found = False
						word_pos_in_sentence = 0

			pageNum += 1

		self.UpdateTextrankScore(articleCount)

		print('Найдено {0} статей'.format(len(self.articles)))
		return self.words, self.lines, self.articles

	def getArticleText(self, articleID, removeNotCyrilicSymbols=True):    
		if (self.lines is None or len(self.lines) == 0):
			raise Exception('Lines are empty. Make sure the parse method was called before')

		if (articleID is None or not isinstance(articleID, int) or int(articleID) < 0):
			return Exception('Argument articleID is incorrect')
		else:
			txt = ''.join([x['text'] for x in self.lines if x['article_num'] == articleID])
			if (removeNotCyrilicSymbols):
				cyrillic = ''.join(re.findall(u"[\u0400-\u0500\(\)\s.,?!:;-]+", txt))
				txt = cyrillic
			return txt


	"""
	Return articles text dataframe for statistical measurement
	"""
	def getArticlesDataframe(self):
		column_names = ['Article No.', 'Author', 
				  'University', 'E-mail', 'Title',
				 'Keywords', 'Source Text']#, 'Preprocessed Text']
		df = pd.DataFrame(columns = column_names)
		
		for i in tqdm(range(len(self.articles))):
			text = self.getArticleText(self.articles[i][0])
			row_df = pd.DataFrame([[self.articles[i][0], 
									self.articles[i][1], 
									self.articles[i][2], 
									self.articles[i][3], 
									self.articles[i][4], 
									self.articles[i][5], 
									text]], columns=column_names)
									#preprocessor.preprocess_text(text)]], columns=column_names)
			df = pd.concat([row_df, df], ignore_index=True)
		return df
	

	"""
	Count textrank score for words when article ends
	"""
	def UpdateTextrankScore(self, articleID):
		if self.generateWordList == True:
			if (len(self.articles) > 0) and len(self.lines) > 0:
				text = ''.join([x['text'] for x in self.lines if x['article_num'] == articleID])
				values = textrank.keywords(text, 
					language='russian', 
					additional_stopwords=self.stopwords, 
					scores=True)
				normalized_values = [self.morph.parse(v[0])[0].inflect({'sing', 'masc', 'nomn'}).word \
						if self.morph.parse(v[0])[0].tag.POS in ['ADJF', 'ADJS', 'PRTF', 'PRTS'] \
						else self.morph.parse(v[0])[0].normal_form.strip() for v in values]

				for w in self.words:
					if (w['article_num'] == articleID and w['morph_normalform'] in normalized_values):
						i = normalized_values.index(w['morph_normalform'])
						w['textrank_score'] = values[i][1]