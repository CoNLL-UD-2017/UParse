import os


ud_vocab = '/home/s1459234/data/ud_vocab/en.vocab'
emb_file = '/home/s1459234/data/embeddings/fastText/wiki.en.vec'


vocab = {}
with open(ud_vocab) as f:
	for line in f:
		word = line.strip()
		vocab[word] = 1

fout = open('/home/s1459234/data/embeddings/projected/en.vec', 'w')
with open(emb_file) as f:
	for line in f:
		line = line.strip()
		fields = line.split()
		word = fields[0]
		if word in vocab:
			fout.write(line + '\n')
