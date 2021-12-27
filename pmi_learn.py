#!python3
#coding=utf-8
#This program take useful data in database to update pmi data 
#==================================================================
#
#Usage:
#  pmi_learn.py [option] [parameter]
#
#	-l
#		list current pmi data
#
#	-h, --help 
#		display this help
#
#==================================================================
#2017/10/02 edit by 陳鐿文
#version 1.0
###########################################################################
import jieba, sys, logging, pickle, math, re, string#, MySQLdb
from jieba import posseg
from collections import defaultdict
import pymysql as MySQLdb




#############################--main function--#############################
def main(mode,num_limit=30):
	###Logging config
	logging.basicConfig(level = logging.DEBUG, handlers = [])
	logger = logging.getLogger('emotion_classify.pmi_learn')
	formatter = logging.Formatter('%(asctime)s %(name)s | line %(lineno)03d: %(levelname)-8s %(message)s','%Y%m%d %H:%M:%S')
	file_handler = logging.FileHandler('./pmi_learn.log', 'a', 'utf-8')
	file_handler.setFormatter(formatter)
	file_handler.setLevel(logging.WARNING)
	formatter = logging.Formatter('%(asctime)s | %(levelname)-8s %(message)s','%Y%m%d %H:%M:%S')
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	stream_handler.setLevel(logging.DEBUG)
	logger.addHandler(file_handler)
	logger.addHandler(stream_handler)

	if mode == "-l":
		#print current pmi data
		pmi_dict=open("./pmi_dict/pmi.pkl","rb")
		i=0
		print("================================================================================")
		print('%-10s%-10s%-15s%-14s%s' % ("詞彙","正面出現次數","負面出現次數","正面pmi","負面pmi"))
		print("================================================================================")
		while True:
			try:
				row=pickle.load(pmi_dict)
				if (row[1] + row[2]) < num_limit:
					continue
				print('%-9s\t%8d\t%8d\t%8.4f\t%8.4f' % (row[0],row[1],row[2],row[3],row[4]))
				i+=1
				if i%20==0:
					print("================================================================================")
					print('%-10s%-10s%-15s%-14s%s' % ("詞彙","正面出現次數","負面出現次數","正面pmi","負面pmi"))
					print("================================================================================")
			except EOFError:
				break
		pmi_dict.close()
		return 0

	#load pmi data
	logger.info("Loading pmi data...")
	word_set=set()
	pos_dict = {}
	pos_num = 0
	neg_dict = {}
	neg_num = 0

	pmi_dict = open("./pmi_dict/pmi.pkl","rb")
	while True:
		try:
			row=pickle.load(pmi_dict)
			word_set.add(row[0])
			pos_dict[row[0]] = row[1]
			pos_num += row[1]
			neg_dict[row[0]] = row[2]
			neg_num += row[2]
		except EOFError:
			break
	pmi_dict.close()
	logger.info("PMI Loading complete...")

	#load jieba dictionary
	jieba.set_dictionary('jieba_dict/dict.txt.big')
	stopwordset = set()
	with open('jieba_dict/stopwords.txt','r',encoding='utf8') as sw:
		for line in sw:
			stopwordset.add(line.strip('\n'))

	#load useful data from database
	###MySQL connect config
	logger.info("Database connecting...")
	try:
		db = MySQLdb.connect(host = "localhost", user = "root", password = "abc123", db = "scrapy", charset = 'utf8')
	except MySQLdb.Error:
		logger.error("Database connect failed")
		sys.exit(-1)
	sql='SELECT count(*) FROM `message` WHERE `authentic` IS NOT NULL AND `pmi_usage` IS NULL'
	#sql='SELECT count(*) FROM `message` WHERE 1'
	cursor = db.cursor()
	cursor.execute(sql)

	row = cursor.fetchone()
	if row[0] == 0:
		logger.info("No more new learnable data")
		return 0

	logger.info("Positive data...")
	sql='SELECT `content` FROM `message` WHERE `authentic`=1 AND `pmi_usage` IS NULL'
	cursor.execute(sql)
	for row in cursor:
		words=posseg.cut(row[0])
		for word in words:
			word.word=re.sub('[%s]'%re.escape(string.punctuation),'',re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）|0-9_]+","",word.word))
			if (word.word == " ") or (word.word in stopwordset):
				continue
			pos_num += 1
			if word.word not in word_set:
				word_set.add(word.word)
			if word.word not in pos_dict:
				pos_dict[word.word] = 1
			else:
				pos_dict[word.word] += 1
	pos_dict = defaultdict(lambda: 0, pos_dict)

	logger.info("Negative data...")
	sql='SELECT `content` FROM `message` WHERE `authentic`=-1 AND `pmi_usage` IS NULL'
	cursor.execute(sql)
	for row in cursor:
		words=posseg.cut(row[0])
		for word in words:
			word.word=re.sub('[%s]'%re.escape(string.punctuation),'',re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）|0-9_]+","",word.word))
			if (word.word == " ") or (word.word in stopwordset):
				continue
			neg_num += 1
			if word.word not in word_set:
				word_set.add(word.word)
			if word.word not in neg_dict:
				neg_dict[word.word] = 1
			else:
				neg_dict[word.word] += 1
	neg_dict = defaultdict(lambda: 0, neg_dict)


	logger.info("PMI computing...")
	pmifile=open("pmi_dict/pmi.pkl",'wb')
	for ele in word_set:
		try:
			pos_pmi=pos_dict[ele]*math.log(((pos_dict[ele]*(neg_num+pos_num))/(pos_num*(pos_dict[ele]+neg_dict[ele]))),2)
		except ValueError:
			pos_pmi=0
		try:
			neg_pmi=neg_dict[ele]*math.log(((neg_dict[ele]*(neg_num+pos_num))/(neg_num*(pos_dict[ele]+neg_dict[ele]))),2)
		except ValueError:
			neg_pmi=0
		word_buffer=[ele,pos_dict[ele],neg_dict[ele],pos_pmi,neg_pmi]
		pickle.dump(word_buffer,pmifile,1)
	pmifile.close()

	sql='UPDATE `message` SET `pmi_usage`=1 WHERE `authentic` IS NOT NULL AND `pmi_usage` IS NULL'
	cursor.execute(sql)
	db.commit()
	cursor.close()
	logger.info("Data update complete")


	return 0
###########################################################################
if __name__ == '__main__':
	if len(sys.argv)==2 and (sys.argv[1]=="-h" or sys.argv[1]=="--help"):
		#Help content
		print("Usage:")
		print("	"+sys.argv[0]+" [option] [parameter]")
		print()
		print("	-l ([num])")
		print("\tlist current pmi data")
		print()
		print("	-h, --help ")
		print("\tdisplay this help")
		print()
	elif (len(sys.argv)==2 or len(sys.argv)==3) and sys.argv[1]=="-l":
		if len(sys.argv)==3:
			main(sys.argv[1],int(sys.argv[2]))
		else:
			main(sys.argv[1])
	elif len(sys.argv)==1:
		main("none")
	else:
		print("Invalid argument")
		sys.exit(-1)
	sys.exit(0)