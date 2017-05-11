import os

ori_file = '/home/s1459234/data/ud_conll2017/UD_Indonesian/id-ud-dev.conllu'
predicted_file = '/home/s1459234/parser/dense_parser/conll2017_models/mono_id/label_train/label_train.h5.id.valid.conllx.out'
out_file = '/home/s1459234/parser/dense_parser/conll2017_models/mono_id/label_train/label_train.h5.id.valid.conllx.res'

content = list()
with open(predicted_file) as f:
	content = f.readlines()

fout = open(out_file, 'w')

i = 0
j = 0
root = True
root_pos = -1
with open(ori_file) as f:
	for line in f:
		line = line.strip()
		if line.startswith('#'):
			j += 1
			continue
		elif line == '':
			assert line == content[i].strip()
			i += 1
			j += 1
			root = True
			root_pos = -1
			fout.write(line + '\n')
		else:
			# read the line
			pred_line = content[i].strip()
			tokens = line.split('\t')
			assert len(tokens) == 10
			if '-' in tokens[0]:
				fout.write(line + '\n')
				j += 1
			else:
				tokens = pred_line.split('\t')
				if tokens[7] == 'root':
					if root:
						root = False
						root_pos = tokens[0]
					else:
						tokens[6] = root_pos
						tokens[7] = 'ccomp'
					pred_line = '\t'.join(tokens)
				fout.write(pred_line + '\n')
				i += 1
				j += 1






