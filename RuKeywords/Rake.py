from rake_nltk import Rake
from tqdm import tqdm
from Metrics import evaluate_keywords

def extractKeywords(texts, num=3, metricsCount=False, partMatchesCounted=False, textprocessor=None):
	r = Rake()
	#textprocessor = textprocessor

	texts['rake_keywords'] = ''
	if (metricsCount == True):
		texts['rake_metrics'] = ''

	for index, row in tqdm(texts.iterrows(), total=texts.shape[0]):
		#l = len(row.text)
		r.extract_keywords_from_text(row.text)
		values = r.get_ranked_phrases()[0:num]
		texts.at[index, 'rake_keywords'] = ','.join(values)

		if (metricsCount == True):
			#add count quality
			#groundtruth = row['Keywords'].split(',')
			if textprocessor.useLemmas == True:
				groundtruth = textprocessor.get_normal_form_text(row['Keywords']).split(',')
			else:
				groundtruth = row['Keywords'].split(',')
				
			metrics = evaluate_keywords(values,groundtruth,partMatchesCounted);
			texts.at[index, 'rake_metrics'] = ','.join([str(m) for m in metrics])
