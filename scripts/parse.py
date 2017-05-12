import os
import sys
import json
import subprocess

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


def load_deprel(codedir):
	valid_deprel = {}
	with open(codedir + '/vocab/ud_rel.vocab') as f:
		cnt = 0
		for line in f:
			line = line.strip()
			if line != 'root':
				valid_deprel[cnt] = line
				cnt += 1
	return valid_deprel


def load_depdist(model, codedir):
	dep_dist_dir = os.path.join(codedir, 'ud-dep-dist')
	tb_deps = os.listdir(dep_dist_dir)
	dep_dist = {}
	dep_file = 'UD_Czech'  # default delex model

	if model in tb_deps:
		dep_file = model
	
	with open(os.path.join(dep_dist_dir, dep_file)) as f:
		for line in f:
			cpos, ppos, deprel = line.strip().split()
			dep_dist[(cpos, ppos)] = deprel
	return dep_dist



def load_setup_file(codedir):
	setup_file = os.path.join(codedir, 'vocab/ud_setup.txt')
	ltcode2model = {}
	lcode2model = {}
	model2len = {}
	ltcode2tb = {}
	with open(setup_file) as f:
		for line in f:
			fields = line.strip().split('\t')
			assert len(fields) == 6
			tb_name = fields[1].lower().replace('-', '_')
			model = fields[2]
			maxlen = fields[3]
			lcode = fields[4]
			ltcode = fields[5]
			if model != 'none':
				model2len[model] = maxlen
			if ltcode != 'none':
				ltcode2model[ltcode] = model
				ltcode2tb[ltcode] = tb_name
			if lcode == ltcode:
				lcode2model[lcode] = model
	return model2len, lcode2model, ltcode2model, ltcode2tb


def process(indir, outdir, codedir, runType):
	data_files = load_json(indir)
	deprel_vocab = load_deprel(codedir)
	model2len, lcode2model, ltcode2model, ltcode2tb = load_setup_file(codedir)
	models = os.listdir(os.path.join(codedir, 'conll2017_models'))

	if runType == 'dense_dist_udpipe':
		udpipe_models = {}
		print('Load udpipe models...')
		for fname in os.listdir(os.path.join(codedir, 'udpipe_models')):
			tb_name = 'ud_' +  fname.split('-')[0]
			udpipe_models[tb_name] = fname
			print(tb_name, fname)
		print()

	for filename in data_files:
		useUDpipe = False
		tb_name = ''
		ltcode = data_files[filename]['ltcode']
		lcode = data_files[filename]['lcode']
		outname = data_files[filename]['outfile']
		if ltcode in ltcode2model:
			model = ltcode2model[ltcode]
		elif lcode in lcode2model:
			model = lcode2model[lcode]
		else:
			model = 'Czech-DEL'

		infile = os.path.join(indir, filename)
		inputFile = os.path.join(outdir, 'incleaned-' + filename)
		outputFile = os.path.join(outdir, 'outcleaned-' + outname)
		finalOutFile = os.path.join(outdir, outname)

		# check if we should use udpipe model
		if runType == 'dense_dist_udpipe':
			if ltcode in ltcode2tb:
				tb_name = ltcode2tb[ltcode]
				if tb_name in udpipe_models:
					useUDpipe = True
		else:
			# check model
			if model not in models or model not in model2len:
				model = 'Czech-DEL'
			print('Parse using DeNse, using model', model)

		if not useUDpipe:
			print('Preprocess input...')
			with open(infile) as f:
				word_cnt = 0
				sent_cnt = 0
				maxlen = int(model2len[model])
				fout = open(inputFile, 'w')
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
			print('Finished preprocessing!')

		if runType == 'dense_dist_udpipe' and useUDpipe:
			print('Parse using UDPipe!')
			print('Parse file:', infile)
			udpipe_command = codedir + '/udpipe-1.1.0-bin/bin-linux64/udpipe --input conllu --output conllu --parse udpipe_models/' + udpipe_models[tb_name] + ' ' + infile + ' > ' + finalOutFile
			os.system(udpipe_command)
		else:
			print('Parse file:', inputFile)
			modelPath = codedir + '/conll2017_models/' + model + '/model_0.001.tune.t7'
			classifier = codedir + '/conll2017_models/' + model + '/lbl_lassifier.t7'
			command = 'th ' + codedir + '/dense_multi_parser.lua --modelPath ' + modelPath + ' --classifierPath ' + classifier + ' --input ' + inputFile + ' --output ' + outputFile + ' --mstalg ChuLiuEdmonds'
			os.system(command)
			print('Finished parsing!')

			print('Post-process output..')
			if runType == 'dense':
				postprocess(infile, outputFile, finalOutFile, deprel_vocab)
			else:
				postprocess_dist(infile, outputFile, finalOutFile, model, codedir, deprel_vocab)
			print('Finished post-processing!')


def postprocess(infile, predFile, outFile, deprel_vocab):
	# get lines from prediction file
	pred_lines = list()
	with open(predFile) as f:
		pred_lines = f.readlines()

	# align sentences
	i = 0
	j = 0
	root = True
	root_pos = -1
	fout = open(outFile, 'w')
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


def postprocess_dist(infile, predFile, outFile, model, codedir, deprel_vocab):
	# load dist file
	deprel_dist = load_depdist(model, codedir)
	# get lines from prediction file
	pred_lines = list()
	with open(predFile) as f:
		pred_lines = f.readlines()

	# align sentences
	i = 0
	j = 0
	root = True
	root_pos = -1
	prev_pos = ''
	fout = open(outFile, 'w')
	with open(infile) as f:
		for line in f:
			line = line.strip()
			pred_line = pred_lines[i].strip()
			if line.startswith('#'):
				fout.write(line + '\n')
			elif line == '':
				assert line == pred_line
				# new sentence restart counter
				root = True
				root_pos = -1
				fout.write(line + '\n')
				i += 1
			elif line != '' and pred_line == '':
				tokens = line.split('\t')
				wid = tokens[0]
				if '-' in wid or '.' in wid:
					fout.write(line + '\n')
				else:
					tokens[6] = str(int(wid) - 1)
					key = tuple([tokens[3], prev_pos])
					if key in deprel_dist:
						rel = deprel_dist[key]
					else:
						rand_idx = randint(0, len(deprel_vocab) - 1)
						rel = deprel_vocab[rand_idx]
					tokens[7] = rel
					new_line = '\t'.join(tokens)
					fout.write(new_line + '\n')
					prev_pos = tokens[3]
			else:
				# read line
				tokens = line.split('\t')
				assert len(tokens) == 10
				if '-' in tokens[0] or '.' in tokens[0]:
					fout.write(line + '\n')
					j += 1
				else:
					fields = pred_line.split('\t')
					assert len(fields) == 10
					fields[2] = tokens[2]
					fields[4] = tokens[4]
					fields[5] = tokens[5]
					if fields[6] == '0' and fields[7] != 'root':
						if fields[0] == '1':
							fields[7] = 'root'
							root = False
							root_pos = fields[0]
						else:
							fields[6] = str(int(fields[0]) - 1)
					elif fields[6] == '0' and fields[7] == 'root':
						if root:
							root = False
							root_pos = fields[0]
						else:  # change prediction of there is already a root
							fields[6] = root_pos
							fields[7] = 'ccomp'
					prev_pos = fields[3]
					pred_line = '\t'.join(fields)
					fout.write(pred_line + '\n')
					i += 1
			j += 1


if __name__=="__main__":
  input_dir = sys.argv[1]
  output_dir = sys.argv[2]
  codedir = sys.argv[3]
  runType = sys.argv[4]
  process(input_dir, output_dir, codedir, runType)


