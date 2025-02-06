# -*- coding: utf-8 -*-
import os
import json
import random
from tqdm import tqdm
from kg import construct_kg


random.seed(42)
prompt = '你是一名医疗人工智能助手，专注于提供准确和可靠的医学信息。请直接回答以下医学相关问题，确保回答简洁明了，不需要进行额外的思考或解释。请注意，所有回答应基于最新的医学知识和研究。'


def construct_entity_level_task_classification(input_kg, outfile):
    """ ENTITY LEVEL - ET """
    ent2type = dict()
    etype_set = set()
    for key, val in input_kg.ents_dict_by_name.items():
        ent, ent_type = key.split('@@')
        if ent_type == 'NA':
            continue
        etype_set.add(ent_type)
        ent2type[ent] = ent_type

    qa_list = list()
    for key, val in input_kg.ents_dict_by_name.items():
        ent, ent_type = key.split('@@')
        ent_type = '其他' if ent_type == 'NA' else ent_type
        if len(etype_set) < 5:
            continue
        etype_candidates = random.sample(etype_set - {ent_type}, 4)
        etype_candidates.append(ent_type)
        random.shuffle(etype_candidates)
        rel_options = list()
        answer = ''
        for idx, etype_cand in enumerate(etype_candidates):
            rel_options.append('(%s) %s' % (chr(ord('A') + idx), etype_cand))
            if etype_cand == ent_type:
                answer = chr(ord('A') + idx)
        qa_list.append({
            'question': '请问医学实体“%s”的实体类型是什么？请从下列选项中选出正确答案\n%s' % (ent, '\n'.join(rel_options)),
            'answer': answer,
            'entity_name': ent,
            'entity_id': val.id
        })
    random.shuffle(qa_list)
    with open(outfile, 'w', encoding='utf-8') as f_out:
        json.dump({
            'prompt': prompt,
            'qa-number': len(qa_list),
            'qa-list': qa_list
        }, f_out, ensure_ascii=False, indent=4)


def construct_entity_level_task_clustering(input_kg, outfile):
    """ ENTITY LEVEL - EC """
    type2ent = dict()
    etype_set = set()
    for key, val in input_kg.ents_dict_by_name.items():
        ent, ent_type = key.split('@@')
        if ent_type == 'NA':
            continue
        if not ent_type in type2ent:
            type2ent[ent_type] = list()
        type2ent[ent_type].append(ent)
        etype_set.add(ent_type)

    qa_list = list()
    for key, val in input_kg.ents_dict_by_name.items():
        ent, ent_type = key.split('@@')
        if ent_type == 'NA':
            continue
        if len(etype_set) < 5:
            continue
        # 随机选择四个非 ent_type 的实体
        ent_type_list = list(etype_set - {ent_type})
        random.shuffle(ent_type_list)
        for etype_idx in range(2):
            if len(type2ent[ent_type_list[etype_idx]]) >= 4:
                candidates = random.sample(type2ent[ent_type_list[etype_idx]], 4)
            else:
                continue
            candidates.append(ent)
            rel_options = list()
            answer = ''
            for idx, entity in enumerate(candidates):
                rel_options.append('(%s) %s' % (chr(ord('A') + idx), entity))
                if entity == ent:
                    answer = chr(ord('A') + idx)
            qa_list.append({
                'question': '请问下列医学实体中，哪个医学实体的实体类型与其他医学实体不一致？请从下列选项中选出正确答案\n%s' % '\n'.join(rel_options),
                'answer': answer,
                'entity_name': ent,
                'entity_id': val.id
            })
    random.shuffle(qa_list)
    with open(outfile, 'w', encoding='utf-8') as f_out:
        json.dump({
            'prompt': prompt,
            'qa-number': len(qa_list),
            'qa-list': qa_list
        }, f_out, ensure_ascii=False, indent=4)


def construct_entity_level_task_disambiguation(input_kg, outfile):
    """ ENTITY LEVEL - ED """
    ent_set = set()
    ent_pairs_pos, ent_pairs_neg = list(), list()
    for triple in input_kg.rels_triple_list:
        if triple[1].name not in ['同义词', '英文别称', '英文别名', '别称']:
            continue
        ent_pairs_pos.append((
            '%s@@%s' % (triple[0].name, triple[0].etype),
            '%s@@%s' % (triple[2].name, triple[2].etype)
        ))
        ent_set.add('%s@@%s' % (triple[0].name, triple[0].etype))
        ent_set.add('%s@@%s' % (triple[2].name, triple[2].etype))

    for ent_1, ent_2 in ent_pairs_pos:
        # 从正样本中采样负样本
        ent_2 = random.sample(ent_set - {ent_1, ent_2}, 1)[0]
        ent_pairs_neg.append((ent_1, ent_2))

    qa_list = list()
    for ent_1, ent_2 in ent_pairs_pos:
        ent_1_id = input_kg.ents_dict_by_name[ent_1].id
        ent_2_id = input_kg.ents_dict_by_name[ent_2].id
        ent_1_name, ent_1_type = ent_1.split('@@')
        ent_2_name, ent_2_type = ent_2.split('@@')
        qa_list.append({
            'question': '请问医学实体“%s”和医学实体“%s”是否是同义词？请回答“是”或“否”。' % (ent_1_name, ent_2_name),
            'answer': '是',
            'entity_1_name': ent_1_name,
            'entity_1_id': ent_1_id,
            'entity_2_name': ent_2_name,
            'entity_2_id': ent_2_id
        })
    for ent_1, ent_2 in ent_pairs_neg:
        ent_1_id = input_kg.ents_dict_by_name[ent_1].id
        ent_2_id = input_kg.ents_dict_by_name[ent_2].id
        ent_1_name, ent_1_type = ent_1.split('@@')
        ent_2_name, ent_2_type = ent_2.split('@@')
        qa_list.append({
            'question': '请问医学实体“%s”和医学实体“%s”是否是同义词？请回答“是”或“否”。' % (ent_1_name, ent_2_name),
            'answer': '否',
            'entity_1_name': ent_1_name,
            'entity_1_id': ent_1_id,
            'entity_2_name': ent_2_name,
            'entity_2_id': ent_2_id
        })
    random.shuffle(qa_list)
    with open(outfile, 'w', encoding='utf-8') as f_out:
        json.dump({
            'prompt': prompt,
            'qa-number': len(qa_list),
            'qa-list': qa_list
        }, f_out, ensure_ascii=False, indent=4)


def construct_relation_level_task_fact_check(input_kg, outfile):
    """ RELATION LEVEL - FC """
    ent_set, rel_set = set(), set()
    fact_list_pos, fact_list_neg = list(), list()
    for triple in input_kg.rels_triple_list:
        fact_list_pos.append((
            '%s@@%s' % (triple[0].name, triple[0].etype),
            triple[1].name,
            '%s@@%s' % (triple[2].name, triple[2].etype)
        ))
        ent_set.add('%s@@%s' % (triple[0].name, triple[0].etype))
        rel_set.add(triple[1].name)
        ent_set.add('%s@@%s' % (triple[2].name, triple[2].etype))

    for head, rel, tail in fact_list_pos:
        # 从正样本中采样负样本
        rel_neg = random.sample(rel_set - {rel}, 1)[0]
        fact_list_neg.append((head, rel_neg, tail))
        tail_neg = random.sample(ent_set - {head, tail}, 1)[0]
        fact_list_neg.append((head, rel, tail_neg))

    qa_list = list()
    for head, rel, tail in fact_list_pos:
        head_id = input_kg.ents_dict_by_name[head].id
        tail_id = input_kg.ents_dict_by_name[tail].id
        rel_id = input_kg.rels_dict_by_name[rel].id
        head_name, head_type = head.split('@@')
        tail_name, tail_type = tail.split('@@')
        qa_list.append({
            'question': '请问医学实体“%s”和医学实体“%s”之间是否存在医学关系“%s”？请回答“是”或“否”。' % (head_name, tail_name, rel),
            'answer': '是',
            'head_name': head_name,
            'head_id': head_id,
            'tail_name': tail_name,
            'tail_id': tail_id,
            'rel_name': rel,
            'rel_id': rel_id
        })
    for head, rel, tail in fact_list_neg:
        head_id = input_kg.ents_dict_by_name[head].id
        tail_id = input_kg.ents_dict_by_name[tail].id
        rel_id = input_kg.rels_dict_by_name[rel].id
        head_name, head_type = head.split('@@')
        tail_name, tail_type = tail.split('@@')
        qa_list.append({
            'question': '请问“%s”和“%s”之间是否存在医学关系“%s”？请回答“是”或“否”。' % (head_name, tail_name, rel),
            'answer': '否',
            'head_name': head_name,
            'head_id': head_id,
            'tail_name': tail_name,
            'tail_id': tail_id,
            'rel_name': rel,
            'rel_id': rel_id
        })
    random.shuffle(qa_list)
    with open(outfile, 'w', encoding='utf-8') as f_out:
        json.dump({
            'prompt': prompt,
            'qa-number': len(qa_list),
            'qa-list': qa_list
        }, f_out, ensure_ascii=False, indent=4)


def construct_relation_level_task_typing(input_kg, outfile):
    """ RELATION LEVEL - RT """
    etype_set, rel_set = set(), set()
    fact_list = list()
    rel2type = dict()
    for triple in input_kg.rels_triple_list:
        if triple[0].etype == 'NA' or triple[2].etype == 'NA':
            continue
        fact_list.append((
            '%s@@%s' % (triple[0].name, triple[0].etype),
            triple[1].name,
            '%s@@%s' % (triple[2].name, triple[2].etype)
        ))
        etype_set.add(triple[0].etype)
        etype_set.add(triple[2].etype)
        rel_set.add(triple[1].name)
        if not triple[1].name in rel2type:
            rel2type[triple[1].name] = (triple[0].etype, triple[2].etype)

    qa_list = list()
    for head, rel, tail in fact_list:
        rel_id = input_kg.rels_dict_by_name[rel].id
        _, head_type = head.split('@@')
        _, tail_type = tail.split('@@')
        if head_type == 'NA' or tail_type == 'NA':
            continue
        etype_candidates = list()
        etype_candidates.append((head_type, tail_type))
        if len(etype_set) < 5:
            continue
        head_type_neg = random.sample(etype_set - {head_type}, 4)
        for jdx in range(4):
            etype_candidates.append((head_type_neg[jdx], tail_type))
        random.shuffle(etype_candidates)
        etype_candidates = ['%s、%s' % (etype_cand[0], etype_cand[1]) for etype_cand in etype_candidates]
        etype_options = list()
        answer = ''
        for idx, etype_cand in enumerate(etype_candidates):
            etype_options.append('(%s) %s' % (chr(ord('A') + idx), etype_cand))
            if etype_cand == '%s、%s' % (head_type, tail_type):
                answer = chr(ord('A') + idx)
        qa_list.append({
            'question': '请问下列医学关系“%s”能连接的头实体类型和尾实体类型组合中哪项是正确的？请从下列选项中选出答案\n%s' % (rel, '\n'.join(etype_options)),
            'answer': answer,
            'rel_name': rel,
            'rel_id': rel_id
        })
    random.shuffle(qa_list)
    with open(outfile, 'w', encoding='utf-8') as f_out:
        json.dump({
            'prompt': prompt,
            'qa-number': len(qa_list),
            'qa-list': qa_list
        }, f_out, ensure_ascii=False, indent=4)


def construct_relation_level_task_rel_completion(input_kg, outfile):
    """ RELATION LEVEL - RP """
    ent_set, rel_set = set(), set()
    fact_list = list()
    for triple in input_kg.rels_triple_list:
        fact_list.append((
            '%s@@%s' % (triple[0].name, triple[0].etype),
            triple[1].name,
            '%s@@%s' % (triple[2].name, triple[2].etype)
        ))
        ent_set.add('%s@@%s' % (triple[0].name, triple[0].etype))
        rel_set.add(triple[1].name)
        ent_set.add('%s@@%s' % (triple[2].name, triple[2].etype))

    qa_list = list()
    for head, rel, tail in fact_list:
        head_id = input_kg.ents_dict_by_name[head].id
        tail_id = input_kg.ents_dict_by_name[tail].id
        rel_id = input_kg.rels_dict_by_name[rel].id
        head_name, head_type = head.split('@@')
        tail_name, tail_type = tail.split('@@')
        rel_candidates = random.sample(rel_set - {rel}, 4)
        rel_candidates.append(rel)
        random.shuffle(rel_candidates)
        rel_options = list()
        answer = ''
        for idx, rel_cand in enumerate(rel_candidates):
            rel_options.append('(%s) %s' % (chr(ord('A') + idx), rel_cand))
            if rel_cand == rel:
                answer = chr(ord('A') + idx)
        qa_list.append({
            'question': '请问医学实体“%s”和医学实体“%s”之间的医学关系是什么？请从下列选项中选出正确答案\n%s' % (head_name, tail_name, '\n'.join(rel_options)),
            'answer': answer,
            'head_name': head_name,
            'head_id': head_id,
            'tail_name': tail_name,
            'tail_id': tail_id,
            'rel_name': rel,
            'rel_id': rel_id
        })
    random.shuffle(qa_list)
    with open(outfile, 'w', encoding='utf-8') as f_out:
        json.dump({
            'prompt': prompt,
            'qa-number': len(qa_list),
            'qa-list': qa_list
        }, f_out, ensure_ascii=False, indent=4)


def construct_subgraph_level_task_fact_check(input_kg, outfile):
    """ SUBGRAPH LEVEL - FC """
    ent_set, rel_set = set(), set()
    fact_list_pos, fact_list_neg = list(), list()
    for triple in input_kg.rels_triple_list:
        fact_list_pos.append((
            '%s@@%s' % (triple[0].name, triple[0].etype),
            triple[1].name,
            '%s@@%s' % (triple[2].name, triple[2].etype)
        ))
        ent_set.add('%s@@%s' % (triple[0].name, triple[0].etype))
        rel_set.add(triple[1].name)
        ent_set.add('%s@@%s' % (triple[2].name, triple[2].etype))

    for head, rel, tail in fact_list_pos:
        # 从正样本中采样负样本
        rel_neg = random.sample(rel_set - {rel}, 1)[0]
        fact_list_neg.append((head, rel_neg, tail))
        tail_neg = random.sample(ent_set - {head, tail}, 1)[0]
        fact_list_neg.append((head, rel, tail_neg))

    random.shuffle(fact_list_pos)
    random.shuffle(fact_list_neg)

    qa_list = list()
    for idx in range(len(fact_list_neg)):
        # sample 4 positive samples
        pos_samples = random.sample(fact_list_pos, 4)
        neg_sample = fact_list_neg[idx]
        # 判定核心是 neg_sample
        head, rel, tail = neg_sample
        head_id = input_kg.ents_dict_by_name[head].id
        tail_id = input_kg.ents_dict_by_name[tail].id
        head_name, head_type = head.split('@@')
        tail_name, tail_type = tail.split('@@')
        rel_id = input_kg.rels_dict_by_name[rel].id
        all_samples = pos_samples + [neg_sample]
        random.shuffle(all_samples)
        rel_options = list()
        answer = ''
        for idx, sample in enumerate(all_samples):
            rel_options.append('(%s) “%s”的“%s”是“%s”' % (chr(ord('A') + idx), sample[0].split('@@')[0], sample[1], sample[2].split('@@')[0]))
            if sample == neg_sample:
                answer = chr(ord('A') + idx)
        qa_list.append({
            'question': '请问下列五个医学关系中错误的医学关系是哪个？请从下列选项中选出错误的医学关系\n%s' % ('\n'.join(rel_options)),
            'answer': answer,
            'head_name': head_name,
            'head_id': head_id,
            'tail_name': tail_name,
            'tail_id': tail_id,
            'rel_name': rel,
            'rel_id': rel_id
        })
    random.shuffle(qa_list)
    with open(outfile, 'w', encoding='utf-8') as f_out:
        json.dump({
            'prompt': prompt,
            'qa-number': len(qa_list),
            'qa-list': qa_list
        }, f_out, ensure_ascii=False, indent=4)


def construct_subgraph_level_task_reasoning(input_kg, outfile_bool, outfile_mcq):
    """ SUBGRAPH LEVEL - R1 & R2 """
    fact_list_pos, fact_list_neg = list(), list()
    for triple in tqdm(input_kg.rels_triple_list):
        head, rel, tail = triple
        flag_pos, flag_neg = False, False
        for rel_1hop, tail_set_1hop in head.involved_as_head_dict.items():
            if flag_pos and flag_neg:
                continue
            for tail_1hop in tail_set_1hop:
                if flag_pos and flag_neg:
                    continue
                for rel_2hop, tail_set_2hop in tail_1hop.involved_as_head_dict.items():
                    if flag_pos and flag_neg:
                        continue
                    for tail_2hop in tail_set_2hop:
                        if flag_pos and flag_neg:
                            continue
                        # check tail2_hop is in tail_set_1hop
                        for rel_temp, tail_set_temp in head.involved_as_head_dict.items():
                            if flag_pos and flag_neg:
                                continue
                            for tail_temp in tail_set_temp:
                                if flag_pos and flag_neg:
                                    continue
                                elif tail_2hop == tail_temp:
                                    fact_list_pos.append({
                                        '1hop': (
                                            '%s@@%s' % (head.name, head.etype),
                                            rel_temp.name,
                                            '%s@@%s' % (tail_temp.name, tail_temp.etype)
                                        ),
                                        '2hop': (
                                            '%s@@%s' % (head.name, head.etype),
                                            rel_1hop.name,
                                            '%s@@%s' % (tail_1hop.name, tail_1hop.etype),
                                            rel_2hop.name,
                                            '%s@@%s' % (tail_2hop.name, tail_2hop.etype)
                                        )
                                    })
                                    flag_pos = True
                                else:
                                    fact_list_neg.append({
                                        '1hop': (
                                            '%s@@%s' % (head.name, head.etype),
                                            rel_temp.name,
                                            '%s@@%s' % (tail_temp.name, tail_temp.etype)
                                        ),
                                        '2hop': (
                                            '%s@@%s' % (head.name, head.etype),
                                            rel_1hop.name,
                                            '%s@@%s' % (tail_1hop.name, tail_1hop.etype),
                                            rel_2hop.name,
                                            '%s@@%s' % (tail_2hop.name, tail_2hop.etype)
                                        )
                                    })
                                    flag_neg = True

    random.shuffle(fact_list_neg)
    fact_list_neg = fact_list_neg[:len(fact_list_pos)]

    qa_bool_list = list()
    for fact_dict in fact_list_pos:
        contexts = '“%s”的“%s”是“%s”，“%s”的“%s”是“%s”' % (
            fact_dict['2hop'][0].split('@@')[0], fact_dict['2hop'][1], fact_dict['2hop'][2].split('@@')[0],
            fact_dict['2hop'][2].split('@@')[0], fact_dict['2hop'][3], fact_dict['2hop'][4].split('@@')[0]
        )
        head, rel, tail = fact_dict['1hop']
        head_id = input_kg.ents_dict_by_name[head].id
        tail_id = input_kg.ents_dict_by_name[tail].id
        rel_id = input_kg.rels_dict_by_name[rel].id
        head_name, head_type = head.split('@@')
        tail_name, tail_type = tail.split('@@')
        qa_bool_list.append({
            'question': '如果已知%s，请问问医学实体“%s”和医学实体“%s”之间是否存在医学关系“%s”？请回答“是”或“否”。' % (contexts, head_name, tail_name, rel),
            'answer': '是',
            'head_name': head_name,
            'head_id': head_id,
            'tail_name': tail_name,
            'tail_id': tail_id,
            'rel_name': rel,
            'rel_id': rel_id
        })
    for fact_dict in fact_list_neg:
        contexts = '“%s”的“%s”是“%s”，“%s”的“%s”是“%s”' % (
            fact_dict['1hop'][0].split('@@')[0], fact_dict['1hop'][1], fact_dict['1hop'][2].split('@@')[0],
            fact_dict['2hop'][2].split('@@')[0], fact_dict['2hop'][3], fact_dict['2hop'][4].split('@@')[0]
        )
        head, rel, tail = fact_dict['1hop']
        head_id = input_kg.ents_dict_by_name[head].id
        tail_id = input_kg.ents_dict_by_name[tail].id
        rel_id = input_kg.rels_dict_by_name[rel].id
        head_name, head_type = head.split('@@')
        tail_name, tail_type = tail.split('@@')
        qa_bool_list.append({
            'question': '如果已知%s，请问问医学实体“%s”和医学实体“%s”之间是否存在医学关系“%s”？请回答“是”或“否”。' % (contexts, head_name, tail_name, rel),
            'answer': '否',
            'head_name': head_name,
            'head_id': head_id,
            'tail_name': tail_name,
            'tail_id': tail_id,
            'rel_name': rel,
            'rel_id': rel_id
        })

    random.shuffle(qa_bool_list)
    with open(outfile_bool, 'w', encoding='utf-8') as f_out:
        json.dump({
            'prompt': prompt,
            'qa-number': len(qa_bool_list),
            'qa-list': qa_bool_list
        }, f_out, ensure_ascii=False, indent=4)

    rel_set = set()
    for triple in input_kg.rels_triple_list:
        rel_set.add(triple[1].name)

    qa_mcq_list = list()
    for fact_dict in fact_list_pos:
        contexts = '“%s”的“%s”是“%s”，“%s”的“%s”是“%s”' % (
            fact_dict['2hop'][0].split('@@')[0], fact_dict['2hop'][1], fact_dict['2hop'][2].split('@@')[0],
            fact_dict['2hop'][2].split('@@')[0], fact_dict['2hop'][3], fact_dict['2hop'][4].split('@@')[0]
        )
        head, rel, tail = fact_dict['1hop']
        head_id = input_kg.ents_dict_by_name[head].id
        tail_id = input_kg.ents_dict_by_name[tail].id
        rel_id = input_kg.rels_dict_by_name[rel].id
        head_name, head_type = head.split('@@')
        tail_name, tail_type = tail.split('@@')
        rel_candidates = random.sample(rel_set - {rel}, 4)
        rel_candidates.append(rel)
        random.shuffle(rel_candidates)
        rel_options = list()
        answer = ''
        for idx, rel_cand in enumerate(rel_candidates):
            rel_options.append('(%s) %s' % (chr(ord('A') + idx), rel_cand))
            if rel_cand == rel:
                answer = chr(ord('A') + idx)
        qa_mcq_list.append({
            'question': '如果已知%s，请问问医学实体“%s”和医学实体“%s”之间是否存在什么医学关系？请从下列选项中选出正确答案\n%s' % (contexts, head_name, tail_name, '\n'.join(rel_options)),
            'answer': answer,
            'head_name': head_name,
            'head_id': head_id,
            'tail_name': tail_name,
            'tail_id': tail_id,
            'rel_name': rel,
            'rel_id': rel_id
        })

    random.shuffle(qa_mcq_list)
    with open(outfile_mcq, 'w', encoding='utf-8') as f_out:
        json.dump({
            'prompt': prompt,
            'qa-number': len(qa_mcq_list),
            'qa-list': qa_mcq_list
        }, f_out, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    for scale in ['large', 'small']:
        kg = construct_kg(
            path_rel_triple='../data/CPubMedKG/%s/CPubMed_%s_relations.csv' % (scale, scale),
            path_attr_triple='../data/CPubMedKG/%s/CPubMed_%s_attributes.csv' % (scale, scale),
            print_stats=True
        )
        construct_entity_level_task_classification(
            input_kg=kg,
            outfile='../data/benchmarks/CPubMedKG_%s/CPubMed_entity_level_task_ET.json' % scale
        )
        construct_entity_level_task_clustering(
            input_kg=kg,
            outfile='../data/benchmarks/CPubMedKG_%s/CPubMed_entity_level_task_EC.json' % scale
        )
        construct_entity_level_task_disambiguation(
            input_kg=kg,
            outfile='../data/benchmarks/CPubMedKG_%s/CPubMed_entity_level_task_ED.json' % scale
        )
        construct_relation_level_task_fact_check(
            input_kg=kg,
            outfile='../data/benchmarks/CPubMedKG_%s/CPubMed_relation_level_task_FC.json' % scale
        )
        construct_relation_level_task_typing(
            input_kg=kg,
            outfile='../data/benchmarks/CPubMedKG_%s/CPubMed_relation_level_task_RT.json' % scale
        )
        construct_relation_level_task_rel_completion(
            input_kg=kg,
            outfile='../data/benchmarks/CPubMedKG_%s/CPubMed_relation_level_task_RP.json' % scale
        )
        construct_subgraph_level_task_fact_check(
            input_kg=kg,
            outfile='../data/benchmarks/CPubMedKG_%s/CPubMed_subgraph_level_task_ER.json' % scale
        )
        construct_subgraph_level_task_reasoning(
            input_kg=kg,
            outfile_bool='../data/benchmarks/CPubMedKG_%s/CPubMed_subgraph_level_task_R1.json' % scale,
            outfile_mcq='../data/benchmarks/CPubMedKG_%s/CPubMed_subgraph_level_task_R2.json' % scale
        )
