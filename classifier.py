import jieba
from jieba import posseg
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures
from collections import defaultdict
import pickle, math, operator, os, sys


class Classifier:
	def __init__(self):
		# jieba custom setting.
		jieba.setLogLevel(20)
		jieba.set_dictionary("/".join(os.path.realpath(__file__).split("/")[:-1]) + '/jieba_dict/dict.txt.big')
		# load stopwords set
		self.stopwordset = set()
		with open("/".join(os.path.realpath(__file__).split("/")[:-1]) + '/jieba_dict/stopwords.txt','r',encoding='utf8') as sw:
			for line in sw:
				self.stopwordset.add(line.strip('\n'))
		self.max_pmi = 0.0
		self.pos_word_num = 0
		self.neg_word_num = 0
		self.total_word_num = 0
		self.pos_prob_dict = {}
		self.neg_prob_dict = {}
		self.pos_pmi_dict = {}
		self.neg_pmi_dict = {}
		self.load()
		#print("max pmi = "+str(self.max_pmi))

	def load(self):
		feature_wordset = self.feature_pick()
		data_file=open("/".join(os.path.realpath(__file__).split("/")[:-1]) + "/pmi_dict/pmi.pkl","rb")
		
		while True:
			try:
				row = pickle.load(data_file)
				self.pos_pmi_dict[row[0]] = row[3]
				self.neg_pmi_dict[row[0]] = row[4]
				if row[3] > self.max_pmi:
					self.max_pmi = row[3]
				if row[4] > self.max_pmi:
					self.max_pmi = row[4]

				if row[0] in feature_wordset:
					self.pos_prob_dict[row[0]] = row[1] / (row[1] + row[2])
					self.neg_prob_dict[row[0]] = row[2] / (row[1] + row[2])
			except EOFError:
				break
		data_file.close()
		return

	def feature_pick(self):
		word_freq = FreqDist()
		cond_word_freq = ConditionalFreqDist()

		word_num = 0
		data_file=open("/".join(os.path.realpath(__file__).split("/")[:-1]) + "/pmi_dict/pmi.pkl","rb")
		while True:
			try:
				row = pickle.load(data_file)
				word_freq[row[0]] += row[1] + row[2]
				cond_word_freq['pos'][row[0]] += row[1]
				cond_word_freq['neg'][row[0]] += row[2]
				word_num += 1
			except EOFError:
				break
		data_file.close()

		self.pos_word_num = cond_word_freq['pos'].N()
		self.neg_word_num = cond_word_freq['neg'].N()
		self.total_word_num = self.pos_word_num + self.neg_word_num

		word_score = {}
		for word, freq in word_freq.items():
			pos_score = BigramAssocMeasures.chi_sq(cond_word_freq['pos'][word], (freq, self.pos_word_num), self.total_word_num) 
			neg_score = BigramAssocMeasures.chi_sq(cond_word_freq['neg'][word], (freq, self.neg_word_num), self.total_word_num)
			word_score[word] = pos_score + neg_score

		sorted_word=sorted(word_score.items(),key=operator.itemgetter(1),reverse=True)
		feature_wordset = set()
		for i in range(math.floor(word_num*0.3)):
			feature_wordset.add(sorted_word[i][0])
		#print(feature_wordset)
		return feature_wordset

	def classify(self,sentence):

		pos_prob = self.pos_word_num/self.total_word_num
		neg_prob = self.neg_word_num/self.total_word_num
		pos_pmi = 0
		neg_pmi = 0

		for word in posseg.cut(sentence):
			if word.word in self.stopwordset:
				continue
			#print(word.word)
			if self.pos_prob_dict.__contains__(word.word):
				pos_prob *= self.pos_prob_dict[word.word]
				neg_prob *= self.neg_prob_dict[word.word]
			if self.pos_pmi_dict.__contains__(word.word):
				pos_pmi += self.pos_pmi_dict[word.word]
				neg_pmi += self.neg_pmi_dict[word.word]


		if pos_prob > (neg_prob + 0.3) :
			return 1
		elif (pos_prob + 0.3) < neg_prob :
			return -1
		else:
			#print(str(pos_prob)+"\t"+str(neg_prob))
			#print(str(pos_pmi)+"\t"+str(neg_pmi))
			if pos_pmi > (neg_pmi + self.max_pmi*0.04):
				return 1
			elif neg_pmi > (pos_pmi + self.max_pmi*0.04):
				return -1
			else:
				return 0



	def test(self,pos_input,neg_input):
		pos_data_num = 0
		neg_data_num = 0
		pos_pos_num = 0
		pos_neg_num = 0
		neg_neg_num = 0
		neg_pos_num = 0
		result = 0

		with open(pos_input,'r',encoding='utf8') as pin:
			for line in pin:
				pos_data_num += 1
				result = self.classify(line)
				if result == 1:
					pos_pos_num += 1
				elif result == -1:
					pos_neg_num += 1


		with open(neg_input,'r',encoding='utf8') as nin:
			for line in nin:
				neg_data_num += 1
				result = self.classify(line)
				if result == 1:
					neg_pos_num += 1
				elif result == -1:
					neg_neg_num += 1

		print('Accuracy = '+str(float(pos_pos_num + neg_neg_num)/float(pos_data_num + neg_data_num)*100.0)+'%')
		print('Bad classify = '+str(float(pos_neg_num + neg_pos_num)/float(pos_data_num + neg_data_num)*100.0)+'%')
		print('Pos Accuracy = '+str(float(pos_pos_num)/float(pos_data_num)*100.0)+'%')
		print('Neg Accuracy = '+str(float(neg_neg_num)/float(neg_data_num)*100.0)+'%')
		return 0


