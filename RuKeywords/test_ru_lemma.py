import rutokenizer
import rupostagger
import rulemma


class test_ru_lemma: #(object)
	"""description of class"""
	def test(self):
		lemmatizer = rulemma.Lemmatizer()
		lemmatizer.load()

		tokenizer = rutokenizer.Tokenizer()
		tokenizer.load()

		tagger = rupostagger.RuPosTagger()
		tagger.load()

		sent = u'Мяукая, голодные кошки ловят жирненьких хрюнделей'
		tokens = tokenizer.tokenize(sent)
		tags = tagger.tag(tokens)
		lemmas = lemmatizer.lemmatize(tags)
		for word, tags, lemma, *_ in lemmas:
			print(u'{:15}\t{:15}\t{}'.format(word, lemma, tags))
