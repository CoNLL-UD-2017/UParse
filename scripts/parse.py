import os
import sys
import json
import subprocess

curdir = '/home/UParse/parser'
# codedir = '.'

from random import randint

def load_json(input):
	json_file = os.path.join(input, 'metadata.json')
	data_files = {}
	with open(json_file) as dat_file:
		data = json.load(dat_file)
		for d in data:
			infile = d['psegmorfile']
			data_files[infile] = d
	return data_files


def load_setup_file():
	setup_file = codedir + '/vocab/ud_setup.txt'
	ltcode2model = {}
	lcode2model = {}
	model2len = {}
	with open(setup_file) as f:
		for line in f:
			fields = line.strip().split('\t')
			assert len(fields) == 6
			model = fields[2]
			maxlen = fields[3]
			lcode = fields[4]
			ltcode = fields[5]
			if model != 'none':
				model2len[model] = maxlen
			if ltcode != 'none':
				ltcode2model[ltcode] = model
			if lcode == ltcode:
				lcode2model[lcode] = model
	return model2len, lcode2model, ltcode2model


def process(indir, outdir):
	data_files = load_json(indir)
	model2len, lcode2model, ltcode2model = load_setup_file()
	for filename in data_files:
		ltcode = data_files[filename]['ltcode']
		lcode = data_files[filename]['lcode']
		outname = data_files[filename]['outfile']
		if ltcode in ltcode2model:
			model = ltcode2model[ltcode]
		elif lcode in lcode2model:
			model = lcode2model[lcode]
		else:
			model = lcode2model['hsb']

		infile = os.path.join(indir, filename)
		inputFile = os.path.join(outdir, 'incleaned-' + filename)
		outputFile = os.path.join(outdir, 'outcleaned-' + outname)
		fout = open(inputFile, 'w')
		with open(infile) as f:
			word_cnt = 0
			sent_cnt = 0
			maxlen = int(model2len[model])
			for line in f:
				line = line.strip()
				
				if line.startswith('#'):
					continue
				
				if line == '':
					fout.write('\n')
					word_cnt = 0
					sent_cnt += 1
				else:
					tokens = line.split('\t')
					assert len(tokens) == 10
					wid = tokens[0]
					if '-' in wid or '.' in wid or word_cnt >= maxlen:
						continue
					else:
						line = '\t'.join(tokens[:6]) + '\t' + str(word_cnt) + '\t' + '\t'.join(tokens[7:])
						word_cnt += 1
					fout.write(line + '\n')
		fout.close()
		print('Parse file:', inputFile)

		modelPath = codedir + '/conll2017_models/' + model + '/model_0.001.tune.t7'
		classifier = codedir + '/conll2017_models/' + model + '/lbl_lassifier.t7'
		command = 'th ' + codedir + '/dense_multi_parser.lua --modelPath ' + modelPath + ' --classifierPath ' + classifier + ' --input ' + inputFile + ' --output ' + outputFile + ' --mstalg ChuLiuEdmonds'
		os.system(command)

		print('Finished parsing!')


def load_vocab():
	valid_deprel = {}
	with open(codedir + '/vocab/ud_rel.vocab') as f:
		cnt = 0
		for line in f:
			line = line.strip()
			valid_deprel[cnt] = line
			cnt += 1
	return valid_deprel


def generateRandomOutput(infile, outfile):
	deprel_vocab = load_vocab()
	fout = open(outfile, 'w')
	with open(infile) as f:
		i = 0
		for line in f:
			line = line.strip()
			if line == '':
				fout.write(line + '\n')
				i = 0
			else:
				tokens = line.split('\t')
				wid = tokens[0]
				if '-' in wid or '.' in wid:
					fout.write(line + '\t')
				else:
					tokens[6] = str(i)
					if i == '0':
						tokens[7] == 'root'
					else:
						rand_idx = randint(0, len(deprel_vocab) - 1)
						tokens[7] = deprel_vocab[rand_idx]
					fout.write('\t'.join(tokens) + '\n')
				i += 1


def postprocess(indir, outdir):
	print('Start post-processing...')
	data_files = load_json(indir)
	model2len, lcode2model, ltcode2model = load_setup_file()
	deprel_vocab = load_vocab()
	for filename in data_files:
		print(filename)
		infile = os.path.join(indir, filename)
		outname = data_files[filename]['outfile']
		outFile = os.path.join(outdir, outname)

		ltcode = data_files[filename]['ltcode']
		lcode = data_files[filename]['lcode']
		outname = data_files[filename]['outfile']
		
		if ltcode in ltcode2model:
			model = ltcode2model[ltcode]
		elif lcode in lcode2model:
			model = lcode2model[lcode]
		else:
			model = lcode2model['hsb']

		predFile = os.path.join(outdir, 'outcleaned-' + outname)
		fout = open(outFile, 'w')
		
		pred_lines = list()
		with open(predFile) as f:
			pred_lines = f.readlines()

		i = 0
		j = 0
		root = True
		root_pos = -1
		with open(infile) as f:
			for line in f:
				line = line.strip()
				pred_line = pred_lines[i].strip()
				if line.startswith('#'):
					fout.write(line + '\n')
					j += 1
					continue
				elif line == '':
					assert line == pred_line
					i += 1
					j += 1
					root = True
					root_pos = -1
					fout.write(line + '\n')
				elif line != '' and pred_line == '':
					tokens = line.split('\t')
					tokens[6] = str(int(tokens[0]) - 1)
					rand_idx = randint(0, len(deprel_vocab) - 1)
					tokens[7] = deprel_vocab[rand_idx]
					new_line = '\t'.join(tokens)
					fout.write(new_line + '\n')
					j += 1
				else:
					# read line
					tokens = line.split('\t')
					assert len(tokens) == 10
					if '-' in tokens[0]:
						fout.write(line + '\n')
						j += 1
					else:
						fields = pred_line.split('\t')
						assert len(fields) == 10
						fields[5] = tokens[5]
						if fields[7] == 'root':
							if root:
								root = False
								root_pos = fields[0]
							else:  # change prediction of there is already a root
								fields[6] = root_pos
								fields[7] = 'ccomp'
							# print(fields)
						pred_line = '\t'.join(fields)
						fout.write(pred_line + '\n')
						i += 1
						j += 1


if __name__=="__main__":
  input_dir = sys.argv[1]
  output_dir = sys.argv[2]
  #process(input_dir, output_dir)
  postprocess(input_dir, output_dir)


