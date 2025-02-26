# -*- coding: utf-8 -*-
from tqdm import tqdm
from kg import construct_kg


def downsample_kg_data_most_freq(input_kg, outfile_rel, outfile_attr, ent_freq_thres=1000):
    rels_triple_list_sampled = list()
    for triple in tqdm(input_kg.rels_triple_list):
        if triple[0].frequency <= ent_freq_thres or triple[2].frequency <= ent_freq_thres:
            continue
        rels_triple_list_sampled.append((
            '%s@@%s' % (triple[0].name, triple[0].etype),
            triple[1].name,
            '%s@@%s' % (triple[2].name, triple[2].etype)
        ))
    with open(outfile_rel, 'w', encoding='utf-8') as f_out:
        for triple in rels_triple_list_sampled:
            f_out.write('\t'.join(triple) + '\n')
    with open(outfile_attr, 'w', encoding='utf-8') as f_out:
        f_out.write('\n')


if __name__ == "__main__":
    # CPubMedKG
    kg = construct_kg(
        path_rel_triple='../data/CPubMedKG/CPubMed_relations.csv',
        path_attr_triple='../data/CPubMedKG/CPubMed_attributes.csv',
        print_stats=False
    )
    downsample_kg_data_most_freq(
        input_kg=kg,
        outfile_rel='../data/MedKGEval/CPubMed_small_relations.csv',
        outfile_attr='../data/MedKGEval/CPubMed_small_attributes.csv',
        ent_freq_thres=3000
    )
    downsample_kg_data_most_freq(
        input_kg=kg,
        outfile_rel='../data/MedKGEval/CPubMed_large_relations.csv',
        outfile_attr='../data/MedKGEval/CPubMed_large_attributes.csv',
        ent_freq_thres=2000
    )

    # CMeKG
    kg = construct_kg(
        path_rel_triple='../data/CMeKG/CMeKG_relations.csv',
        path_attr_triple='../data/CMeKG/CMeKG_attributes.csv',
        print_stats=False
    )
    downsample_kg_data_most_freq(
        input_kg=kg,
        outfile_rel='../data/MedKGEval/CMeKG_small_relations.csv',
        outfile_attr='../data/MedKGEval/CMeKG_small_attributes.csv',
        ent_freq_thres=400
    )
    downsample_kg_data_most_freq(
        input_kg=kg,
        outfile_rel='../data/MedKGEval/CMeKG_large_relations.csv',
        outfile_attr='../data/MedKGEval/CMeKG_large_attributes.csv',
        ent_freq_thres=300
    )
