# -*- coding: utf-8 -*-
from tqdm import tqdm


def process_CPubMedKG(infile, rel_outfile, attr_outile):
    ent2type = dict()
    triple_list = list()
    with open(infile, 'r', encoding='utf-8') as f_in:
        for line in tqdm(f_in):
            elems = line.strip().split('\t')
            if elems[1] == 'relation':
                continue
            if '@@' in elems[0]:
                head, head_type = elems[0].split('@@')
            else:
                head, head_type = elems[0], ''
            relation = elems[1]
            if '@@' in elems[2]:
                tail, tail_type = elems[2].split('@@')
            else:
                tail, tail_type = elems[2], ''
            if head not in ent2type:
                ent2type[head] = head_type
            elif ent2type[head] == '':
                ent2type[head] = head_type
            if tail not in ent2type:
                ent2type[tail] = tail_type
            elif ent2type[tail] == '':
                ent2type[tail] = tail_type
            triple_list.append([head, head_type, relation, tail, tail_type])
    # process empty entity type
    for idx in tqdm(range(len(triple_list))):
        if triple_list[idx][1] == '' and triple_list[idx][0] in ent2type:
            triple_list[idx][1] = ent2type[triple_list[idx][0]]
        if triple_list[idx][4] == '' and triple_list[idx][3] in ent2type:
            triple_list[idx][4] = ent2type[triple_list[idx][3]]
    # write to file
    with open(rel_outfile, 'w', encoding='utf-8') as f_out:
        for triple in tqdm(triple_list):
            f_out.write('%s@@%s\t%s\t%s@@%s\n' % (
                triple[0], triple[1], triple[2], triple[3], triple[4]
            ))
            f_out.flush()
    with open(attr_outile, 'w', encoding='utf-8') as f_out:
        f_out.write('\n')


def process_CMeKG(infile, rel_outfile, attr_outile):
    ent2type = dict()
    triple_list = list()
    with open(infile, 'r', encoding='utf-8') as f_in:
        for line in tqdm(f_in):
            elems = line.strip().split('\t')
            head = elems[0]
            relation = elems[1]
            tail = elems[2]
            if relation == '分类':
                ent2type[head] = tail
            else:
                triple_list.append([head, '', relation, tail, ''])
    # process empty entity type
    for idx in tqdm(range(len(triple_list))):
        if triple_list[idx][1] == '' and triple_list[idx][0] in ent2type:
            triple_list[idx][1] = ent2type[triple_list[idx][0]]
        if triple_list[idx][4] == '' and triple_list[idx][3] in ent2type:
            triple_list[idx][4] = ent2type[triple_list[idx][3]]
    # write to file
    with open(rel_outfile, 'w', encoding='utf-8') as f_out:
        for triple in tqdm(triple_list):
            f_out.write('%s@@%s\t%s\t%s@@%s\n' % (
                triple[0], triple[1] if triple[1] != '' else 'NA',
                triple[2],
                triple[3], triple[4] if triple[4] != '' else 'NA'
            ))
            f_out.flush()
    with open(attr_outile, 'w', encoding='utf-8') as f_out:
        f_out.write('\n')


if __name__ == "__main__":
    process_CPubMedKG(
        infile='../data/CPubMedKG/CPubMed-KGv1.txt',
        rel_outfile='../data/CPubMedKG/CPubMed_relations.csv',
        attr_outile='../data/CPubMedKG/CPubMed_attributes.csv'
    )
    process_CMeKG(
        infile='../data/CMeKG/CMeKGv1.0.csv',
        rel_outfile='../data/CMeKG/CMeKG_relations.csv',
        attr_outile='../data/CMeKG/CMeKG_attributes.csv'
    )
