import os
from collections import defaultdict

# create cleaned UD datasets
# remove multiword tokens and language specific dep relations
data_dir = '/home/s1459234/data/ud-treebanks-conll2017'
out_data_dir = '/home/s1459234/data/conll2017_data'
group_file = '/home/s1459234/parser/dense_parser/vocab/ud_grouped.txt'


lang_group = {}
current = ''
with open(group_file) as f:
    for line in f:
        line = line.strip()
        if line == '':
            continue
        if line.startswith('Group: '):
            current = line.replace('Group: ', '')
            lang_group[current] = list()
        else:
            lang_group[current].append(line)


for group_name in lang_group:
    print(group_name)
    out_dir = os.path.join(out_data_dir, group_name)
    try:
        os.stat(out_dir)
    except:
        os.mkdir(out_dir)
        os.mkdir(out_dir + '/train')
        os.mkdir(out_dir + '/dev')
    for treebank in lang_group[group_name]:
        tb_dir = os.path.join(data_dir, treebank)
        for file_name in os.listdir(tb_dir):
            if 'train.conllu' in file_name or 'dev.conllu' in file_name:
                print(file_name)
                if 'train' in file_name:
                    tokens = file_name.split('-', 1)
                    new_name = file_name
                    if '_' in tokens[0]:
                        lang = tokens[0].split('_')[0]
                        new_name = lang + '-' + tokens[1]
                    f_out = open(os.path.join(out_dir, 'train/cleaned-' + new_name), 'a')
                else:
                    f_out = open(os.path.join(out_dir, 'dev/cleaned-' + file_name), 'w')
                with open(os.path.join(tb_dir, file_name)) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('#'):
                            continue
                        if line == '':
                            f_out.write(line + '\n')
                        else:
                            tokens = line.split('\t')
                            assert len(tokens) == 10
                            wid = tokens[0]
                            if '-' in wid or '.' in wid:
                                continue
                            else:
                                rel = tokens[7]
                                new_rel = rel.split(':')[0]
                                line = line.replace(rel, new_rel)
                                f_out.write(line + '\n')
                f_out.close()        


