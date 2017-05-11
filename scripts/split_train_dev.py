import os
import math

train_data = '/home/s1459234/data/conll2017_data/Turkic-DEL/train/cleaned-ug-ud-train.conllu'
out_train_data = '/home/s1459234/data/conll2017_data/Turkic-DEL/cleaned-ug-ud-train.conllu'
out_dev_data = '/home/s1459234/data/conll2017_data/Turkic-DEL/cleaned-ug-ud-dev.conllu'

num_sents = 100


train_split = math.floor(num_sents * 0.8)
dev_split = num_sents - train_split
f_train = open(out_train_data, 'w')
f_dev = open(out_dev_data, 'w')

counter = 0
with open(train_data) as f:
	for line in f:
		line = line.strip()
		if counter <= train_split:
			f_train.write(line + '\n')
		else:
			f_dev.write(line + '\n')
		if line == '':
			counter += 1

print(counter)