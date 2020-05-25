from tqdm import tqdm
from Metrics import evaluate_keywords
from summa import keywords as textrank
import random

class TextRankExtractor:

	def __init__(self, stopwordsList=None, language='russian', textprocessor=None, *args, **kwargs):
		self.stopwords = stopwordsList
		self.language = language
		self.textprocessor = textprocessor

	def extractKeywords(self, texts, num=5, metricsCount=False, partMatchesCounted=False):
		texts['textrank_keywords'] = ''

		if (metricsCount == True):
			texts['textrank_metrics'] = ''
			
			#sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
			#print([word[0] for word in sorted_words]
			#values = [word[0] for word in sorted_words[:num]]
			
		for index, row in tqdm(texts.iterrows(), total=texts.shape[0]):
			values = textrank.keywords(row.text, language=self.language, additional_stopwords=self.stopwords)
			values = values.split('\n')
			length = len(values)

			values = ','.join(random.sample(values, min(num, length)))
			texts.at[index, 'textrank_keywords'] = values

			if (metricsCount == True):
				if self.textprocessor.useLemmas == True:
					groundtruth = self.textprocessor.get_normal_form_text(row['Keywords']).split(',')
				else:
					groundtruth = row['Keywords'].split(',')
				
				metrics = evaluate_keywords(values.split(','),groundtruth,partMatchesCounted)
				texts.at[index, 'textrank_metrics'] = ','.join([str(m) for m in metrics])
