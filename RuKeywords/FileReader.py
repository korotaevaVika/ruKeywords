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


class Reader:

	def __init__(self, fileName, generateWordList=False): 	
		self.fileName = fileName
		self.doc = None
		self.pageObjects = []
		self.pages = []
		self.words = []
		self.lines = []
		self.articles = []

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
	def parseDocPages(self, pageStart=0, pageEnd=None, includeRemarks=False):
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

		#tag_dict = {"J": wordnet.ADJ,
		#			"N": wordnet.NOUN,
		#			"V": wordnet.VERB,
		#			"R": wordnet.ADV}

		while (pageNum < pageEnd):
			page = self.doc.loadPage(pageNum)
			dictionary = page.getText("dict")
			block_count = 0
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
					word_count_insentence = 0

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
								author = unicodedata.normalize('NFKC', line['spans'][0]['text'])
								university = ''
								mail = ''
								title = ''
								keywords = ''
								isKeywordsSpan = False
							elif (block_count == 0 and authorsBlock == True and 'Bold' in line['spans'][0]['font']):
								author += unicodedata.normalize('NFKC', line['spans'][0]['text'])
								processSpans = False
								
							if (processSpans == True):
								for span in line['spans']:                        
									if (isNewArticle == True and 'Italic' in span['font'] and int(span['size']) == 10 and mail == ''):
										university += unicodedata.normalize('NFKC', span['text'])

									elif (isNewArticle == True and 'DJGIDG+SchoolBookC' in span['font'] and int(span['size']) == 8):
										mail += unicodedata.normalize('NFKC', span['text'])

									elif (isNewArticle == True and 'Bold' in span['font']):
										title += unicodedata.normalize('NFKC', span['text']).replace('\uf6ba', ' ') #+

									elif (isNewArticle == True and 'Italic' in span['font'] and int(span['size']) == 8 and 'Ключевые слова' in unicodedata.normalize('NFKC', span['text']) and not title == '' and keywords == ''):
										isKeywordsSpan = True

									elif (isNewArticle == True and isKeywordsSpan == True):

										removeDotsWhenKeywordsStart = False
										#remove переносы по слогам
										if (len(keywords) > 0 and keywords[-1] == '-'):
											keywords = keywords[:-1]
										elif (keywords == '' and unicodedata.normalize('NFKC', span['text'])[0] == ':'):
											removeDotsWhenKeywordsStart = True

										keywords+= unicodedata.normalize('NFKC', span['text'])

										if (removeDotsWhenKeywordsStart == True):
											keywords = keywords[1:]

										if (len(unicodedata.normalize('NFKC', span['text']).strip()) > 0 and unicodedata.normalize('NFKC', span['text']).strip()[-1] == '.'):
											articleCount += 1
											isNewArticle = False
											keywords = keywords[:-1].strip() #remove last dot and all useless spaces
											author = author.strip()
											university = university.strip()
											mail = mail.strip()
											title = title.strip()
											self.articles.append([articleCount, author, university, mail, title, keywords])
											author = ''
											university = ''
											mail = ''
											title = ''
											keywords = ''
											isKeywordsSpan = False


									elif (isNewArticle == False and (includeRemarks == False and int(span['size']) > 8) #not example remarks
										  ):   

										lineText = ''

										prevWord = None
										# удаление переноса по слогам в предыдущей строке
										if (span_count == 0 and len(self.lines) > 0 and self.lines[-1]['text'][-1] == '-' and unicodedata.normalize('NFKC', span['text'])[0].islower() == True):
											prevWord = self.words.pop()
											n = len(prevWord['text']) + 1 # +1 на дефис
											self.lines[-1]['text'] = self.lines[-1]['text'][:-n]
											prevWord['text'] = prevWord['text'][:-1]

											if (prevWord != None and len(prevWord['text'][0]) > 0 and not(prevWord['text'][0] == ' ' or prevWord['text'][0] == '')):
												lineText += ' '
											lineText += prevWord['text'] 

										if (('НАПРАВЛЕНИE' in unicodedata.normalize('NFKC', span['text']).upper()) or span['size'] == 13):
											#switch to the next page
											goToNextPage = True
											break

										lineText += unicodedata.normalize('NFKC', span['text'])
										#swapLineText = ''.join(
										#    re.findall(u"[\u0400-\u0500\(\)\s.,?!:;-]+", lineText))
                                    
										#if (not swapLineText == lineText):
										#    print('linetext = "{0}", swap = "{1}"'.format(lineText,
										#    swapLineText))
										#    lineText = str(swapLineText)
                                    
                                    
										self.lines.append({'size':span['size'], 
														  'flags': span['flags'],
														  'font': span['font'],
														  'color': span['color'],
														  'text': lineText, #unicodedata.normalize('NFKC', span['text']),
														  'span_count': span_count,
														  'line_count': line_count,
														  'block_count': block_count,
														  'page_num': pageNum,
														  'article_num': articleCount
													 })

										word_count_inline = 0
										for txt in lineText.split(' '):
											#запишем слово вместе с его слогами с предыдущей строки
											if not prevWord is None and not (unicodedata.normalize('NFKC', txt) in [" ", ""]):
												word = prevWord['text'] + unicodedata.normalize('NFKC', txt)
												prevWord = None
											else:
												word = unicodedata.normalize('NFKC', txt)

											otherSigns = "".join(re.findall("[^а-яА-Яa-zA-Z,-]", word))
											if (not otherSigns == ""): #(regex.match(word)):
												word = "".join(re.findall("[а-яА-Яa-zA-Z,-]+", word))
											
											if (self.generateWordList == True and word == "" and '.' in otherSigns):
												self.words[-1]['otherSigns'] = otherSigns
												dot_found = True

											if (word == ""):
												prevWord = None
												continue

											if (self.generateWordList == False):
												self.words.append({'text': word})
												continue

											word_count_inline = word_count_inline + 1

											x = self.morph.parse(word)[0]

											if dot_found == False and '.' in otherSigns and not '..' in otherSigns:
												dot_found = True
											else:
												if (dot_found and len(word) > 0 and word[0].isupper()
													and (self.words[-1] is not None and self.words[-1]['isupper'] == False)):
													dot_found = False
													print('sentence_count ', sentence_count)
													print('block_count ', block_count)
													print('line_count ', line_count)
													print('page_num ', pageNum)
													print(' '.join([x['text'] for x in self.words if x['article_num'] == articleCount and x['sentence_count'] == sentence_count and x['line_count'] == line_count and x['page_num'] == pageNum and x['block_count'] == block_count]))
													sentence_count = sentence_count + 1
												
												elif dot_found and len(word) > 0 and word[0].islower():
													dot_found = False

											self.words.append({'size':span['size'], 
														  'flags': span['flags'],
														  'font': span['font'],
														  'color': span['color'],
														  'otherSigns': otherSigns,
														  'istitle': word[0:2].istitle(),
														  'isupper': word.isupper(),
														  'text': word, #unicodedata.normalize('NFKC', txt),

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
														  'morph_normalform' : x.normal_form, # неопределенная форма
														  
														  'word_count_inline': word_count_inline,
														  'span_count': span_count,
														  'line_count': line_count,
														  'block_count': block_count,
														  'sentence_count': sentence_count,
														  'page_num': pageNum,
														  'article_num': articleCount
													 })
										span_count += 1
							line_count += 1
							

							if goToNextPage == True:
								break

						if goToNextPage == True:
							break
						block_count += 1

						sentence_count = 0
						dot_found = False
			pageNum += 1
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
