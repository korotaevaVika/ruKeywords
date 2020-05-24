from tqdm import tqdm
from Metrics import evaluate_keywords
import math
from textblob import TextBlob as tb

class TfIdfBlobExtractor:

	def __init__(self, stopwordsList=None, textprocessor=None, *args, **kwargs):
		self.stopwords = stopwordsList
		self.textprocessor = textprocessor

	def tf(self, word, blob):
		if word in self.stopwords:
			return 0;
		else:
			return blob.words.count(word) / len(blob.words)

	def n_containing(self, word, bloblist):
		return sum(1 for blob in bloblist if word in blob.words)

	def idf(self, word, bloblist):
		return math.log(len(bloblist) / (1 + self.n_containing(word, bloblist)))

	def tfidf(self, word, blob, bloblist):
		return self.tf(word, blob) * self.idf(word, bloblist)

	def extractKeywords(self, texts, num=5, metricsCount=False, partMatchesCounted=False):
		texts['tfidf_blob_keywords'] = ''

		if (metricsCount == True):
			texts['tfidf_blob_metrics'] = ''

		bloblist = [tb(x) for x in texts.text.values]
		
		for i, row in tqdm(texts.iterrows(), total=texts.shape[0]):
		#for i in tqdm(range(len(bloblist))):
			blob = bloblist[i]
			scores = {word: self.tfidf(word, blob, bloblist) for word in blob.words}
			sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
			#print([word[0] for word in sorted_words]
			values = [word[0] for word in sorted_words[:num]]
			texts.at[i, 'tfidf_blob_keywords'] = ','.join(values)	

			if (metricsCount == True):
				#add count quality
				#groundtruth = row['Keywords'].split(',')
				if self.textprocessor.useLemmas == True:
					groundtruth = self.textprocessor.get_normal_form_text(row['Keywords']).split(',')
				else:
					groundtruth = row['Keywords'].split(',')
				
				metrics = evaluate_keywords(values,groundtruth,partMatchesCounted);
				texts.at[i, 'tfidf_blob_metrics'] = ','.join([str(m) for m in metrics])
