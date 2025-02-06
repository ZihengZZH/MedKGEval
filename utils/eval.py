# -*- coding: utf-8 -*-
import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from kg import construct_kg


def check_equal_for_cls(ans_1, ans_2):
    if ans_1 == ans_2:
        return True
    if ans_1 == '是' and ans_2 in ['是', '是。', '是的', '是的。']:
        return True
    if ans_1 == '否' and ans_2 in ['否', '否。', '不', '不是', '不是的', '不是的。']:
        return True
    if ans_1 == '是' and ans_2.startswith('是'):
        return True
    if ans_1 == '否' and ans_2.startswith('不'):
        return True
    return False


def check_equal_for_mcq(ans_1, ans_2):
    # extract the first A/B/C/D/E answer from ans_2
    ans_2 = re.sub(r'[^A-E]', '', ans_2)
    if ans_1 == ans_2:
        return True
    return False


def evaluate_cls(infile):
    ent_id2result = dict()
    rel_id2result = dict()
    with open(infile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    correct, total = 0, 0
    tokens_all, time_all = 0, 0.0
    for item in data:
        if check_equal_for_cls(item['answer'], item['generation']):
            correct += 1
            if item.__contains__('entity_id'):
                if not ent_id2result.__contains__(item['entity_id']):
                    ent_id2result[item['entity_id']] = list()
                ent_id2result[item['entity_id']].append(1)
            if item.__contains__('entity_1_id'):
                if not ent_id2result.__contains__(item['entity_1_id']):
                    ent_id2result[item['entity_1_id']] = list()
                ent_id2result[item['entity_1_id']].append(1)
            if item.__contains__('entity_2_id'):
                if not ent_id2result.__contains__(item['entity_2_id']):
                    ent_id2result[item['entity_2_id']] = list()
                ent_id2result[item['entity_2_id']].append(1)
            if item.__contains__('head_id'):
                if not ent_id2result.__contains__(item['head_id']):
                    ent_id2result[item['head_id']] = list()
                ent_id2result[item['head_id']].append(1)
            if item.__contains__('tail_id'):
                if not ent_id2result.__contains__(item['tail_id']):
                    ent_id2result[item['tail_id']] = list()
                ent_id2result[item['tail_id']].append(1)
            if item.__contains__('rel_id'):
                if not rel_id2result.__contains__(item['rel_id']):
                    rel_id2result[item['rel_id']] = list()
                rel_id2result[item['rel_id']].append(1)
        else:
            if item.__contains__('entity_id'):
                if not ent_id2result.__contains__(item['entity_id']):
                    ent_id2result[item['entity_id']] = list()
                ent_id2result[item['entity_id']].append(0)
            if item.__contains__('entity_1_id'):
                if not ent_id2result.__contains__(item['entity_1_id']):
                    ent_id2result[item['entity_1_id']] = list()
                ent_id2result[item['entity_1_id']].append(0)
            if item.__contains__('entity_2_id'):
                if not ent_id2result.__contains__(item['entity_2_id']):
                    ent_id2result[item['entity_2_id']] = list()
                ent_id2result[item['entity_2_id']].append(0)
            if item.__contains__('head_id'):
                if not ent_id2result.__contains__(item['head_id']):
                    ent_id2result[item['head_id']] = list()
                ent_id2result[item['head_id']].append(0)
            if item.__contains__('tail_id'):
                if not ent_id2result.__contains__(item['tail_id']):
                    ent_id2result[item['tail_id']] = list()
                ent_id2result[item['tail_id']].append(0)
            if item.__contains__('rel_id'):
                if not rel_id2result.__contains__(item['rel_id']):
                    rel_id2result[item['rel_id']] = list()
                rel_id2result[item['rel_id']].append(0)
        total += 1
        tokens_all += item['tokens']
        time_all += float(item['timecost'])
    print(infile)
    if total != 0:
        acc = correct / total
        print('Accuracy: %.4f' % acc)
    else:
        acc = 0.0
        print('Accuracy: 0.0000')
    if time_all != 0.0:
        print('Avg Tokens/s: %.4f' % (tokens_all / time_all))
    else:
        print('Avg Tokens/s: 0.0000')
    return ent_id2result, rel_id2result, acc


def evaluate_mcq(infile):
    ent_id2result = dict()
    rel_id2result = dict()
    with open(infile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    correct, total = 0, 0
    tokens_all, time_all = 0, 0.0
    for item in data:
        if check_equal_for_mcq(item['answer'], item['generation']):
            correct += 1
            if item.__contains__('entity_id'):
                if not ent_id2result.__contains__(item['entity_id']):
                    ent_id2result[item['entity_id']] = list()
                ent_id2result[item['entity_id']].append(1)
            if item.__contains__('entity_1_id'):
                if not ent_id2result.__contains__(item['entity_1_id']):
                    ent_id2result[item['entity_1_id']] = list()
                ent_id2result[item['entity_1_id']].append(1)
            if item.__contains__('entity_2_id'):
                if not ent_id2result.__contains__(item['entity_2_id']):
                    ent_id2result[item['entity_2_id']] = list()
                ent_id2result[item['entity_2_id']].append(1)
            if item.__contains__('head_id'):
                if not ent_id2result.__contains__(item['head_id']):
                    ent_id2result[item['head_id']] = list()
                ent_id2result[item['head_id']].append(1)
            if item.__contains__('tail_id'):
                if not ent_id2result.__contains__(item['tail_id']):
                    ent_id2result[item['tail_id']] = list()
                ent_id2result[item['tail_id']].append(1)
            if item.__contains__('rel_id'):
                if not rel_id2result.__contains__(item['rel_id']):
                    rel_id2result[item['rel_id']] = list()
                rel_id2result[item['rel_id']].append(1)
        else:
            if item.__contains__('entity_id'):
                if not ent_id2result.__contains__(item['entity_id']):
                    ent_id2result[item['entity_id']] = list()
                ent_id2result[item['entity_id']].append(0)
            if item.__contains__('entity_1_id'):
                if not ent_id2result.__contains__(item['entity_1_id']):
                    ent_id2result[item['entity_1_id']] = list()
                ent_id2result[item['entity_1_id']].append(0)
            if item.__contains__('entity_2_id'):
                if not ent_id2result.__contains__(item['entity_2_id']):
                    ent_id2result[item['entity_2_id']] = list()
                ent_id2result[item['entity_2_id']].append(0)
            if item.__contains__('head_id'):
                if not ent_id2result.__contains__(item['head_id']):
                    ent_id2result[item['head_id']] = list()
                ent_id2result[item['head_id']].append(0)
            if item.__contains__('tail_id'):
                if not ent_id2result.__contains__(item['tail_id']):
                    ent_id2result[item['tail_id']] = list()
                ent_id2result[item['tail_id']].append(0)
            if item.__contains__('rel_id'):
                if not rel_id2result.__contains__(item['rel_id']):
                    rel_id2result[item['rel_id']] = list()
                rel_id2result[item['rel_id']].append(0)
        total += 1
        tokens_all += item['tokens']
        time_all += float(item['timecost'])
    print(infile)
    if total != 0:
        acc = correct / total
        print('Accuracy: %.4f' % acc)
    else:
        acc = 0.0
        print('Accuracy: 0.0000')
    if time_all != 0.0:
        print('Avg Tokens/s: %.4f' % (tokens_all / time_all))
    else:
        print('Avg Tokens/s: 0.0000')
    return ent_id2result, rel_id2result, acc


def compute_coverage_ratio_avg(id2result_list):
    """ Averaged Coverage """
    id2ratio = dict()
    for id2result in id2result_list:
        for ent_id, res in id2result.items():
            if not id2ratio.__contains__(ent_id):
                id2ratio[ent_id] = list()
            id2ratio[ent_id].append(sum(res) / len(res))
    for id, ratio in id2ratio.items():
        id2ratio[id] = np.mean(ratio)
    print('Coverage: %.4f' % (np.mean(list(id2ratio.values()))))
    return id2ratio


def compute_coverage_ratio_degree_aware(id2result_list, input_kg, aspect='ent'):
    """ Degree-aware Coverage """
    id2ratio = dict()
    for id2result in id2result_list:
        for ent_id, res in id2result.items():
            if not id2ratio.__contains__(ent_id):
                id2ratio[ent_id] = list()
            id2ratio[ent_id].append(sum(res) / len(res))
    for id, ratio in id2ratio.items():
        id2ratio[id] = np.mean(ratio)

    value = 0.0

    if aspect == 'ent':
        deg_all = 0
        for ent_id, ratio in id2ratio.items():
            entity = input_kg.ents_dict_by_id[ent_id]
            deg_as_head = 0
            for _, tail_list in entity.involved_as_head_dict.items():
                deg_as_head += len(tail_list)
            deg_as_tail = 0
            for _, head_list in entity.involved_as_tail_dict.items():
                deg_as_tail += len(head_list)
            deg = deg_as_head + deg_as_tail
            deg_all += deg
            value += ratio * deg / (len(input_kg.rels_triple_list) * 2)
        if deg_all == len(input_kg.rels_triple_list) * 2:
            print('Coverage: %.4f' % value)
        else:
            print('Coverage: nan')

    elif aspect == 'rel':
        deg_all = 0
        for rel_id, ratio in id2ratio.items():
            relation = input_kg.rels_dict_by_id[rel_id]
            deg = relation.frequency
            deg_all += deg
            value += ratio * deg / len(input_kg.rels_triple_list)
        if deg_all == len(input_kg.rels_triple_list):
            print('Coverage: %.4f' % value)
        else:
            print('Coverage: nan')

    return id2ratio


def compute_coverage_triple(ent_id2result_list, rel_id2result_list, input_kg):
    ent_id2ratio = dict()
    for ent_id2result in ent_id2result_list:
        for ent_id, res in ent_id2result.items():
            if not ent_id2ratio.__contains__(ent_id):
                ent_id2ratio[ent_id] = list()
            ent_id2ratio[ent_id].append(sum(res) / len(res))
    for id, ratio in ent_id2ratio.items():
        ent_id2ratio[id] = np.mean(ratio)
    rel_id2ratio = dict()
    for rel_id2result in rel_id2result_list:
        for rel_id, res in rel_id2result.items():
            if not rel_id2ratio.__contains__(rel_id):
                rel_id2ratio[rel_id] = list()
            rel_id2ratio[rel_id].append(sum(res) / len(res))
    for id, ratio in rel_id2ratio.items():
        rel_id2ratio[id] = np.mean(ratio)
    triple_id2ratio = dict()
    triple_id2name = dict()
    for idx, triple in enumerate(input_kg.rels_triple_list):
        if (ent_id2ratio.__contains__(triple[0].id) and
            rel_id2ratio.__contains__(triple[1].id) and
            ent_id2ratio.__contains__(triple[2].id)):
            triple_id2ratio[idx] = np.mean([
                ent_id2ratio[triple[0].id],
                rel_id2ratio[triple[1].id],
                ent_id2ratio[triple[2].id]
            ])
            triple_id2name[idx] = '%s\t%s\t%s' % (
                input_kg.ents_dict_by_id[triple[0].id].name,
                input_kg.rels_dict_by_id[triple[1].id].name,
                input_kg.ents_dict_by_id[triple[2].id].name
            )

    # # 按照正确率排序倒序打印前10个
    # for idx, (triple_id, ratio) in enumerate(sorted(triple_id2ratio.items(), key=lambda x: x[1], reverse=True)):
    #     if idx < 10:
    #         print('%s\t%.4f' % (triple_id2name[triple_id], ratio))

    print('Coverage: %.4f' % (np.mean(list(triple_id2ratio.values()))))
    return triple_id2ratio


if __name__ == "__main__":

    # Task-Level Evaluation

    llm_names = [
        'qwen2-0.5b',
        'qwen2-1.5b',
        'qwen2-7b',
        # 'baichuan2-7b',
        # 'baichuan2-13b',
        # 'medllm',
        'huatuogpt2-7b',
        'huatuogpt2-13b',
        # 'pulse',
        # 'wingpt2',
        # 'gpt4o'
    ]

    kg_names = ['CPubMedKG']
    scales = ['small']

    for kg_name in kg_names:

        acc_dict = dict()
        ent_id2result_all = dict()
        rel_id2result_all = dict()

        for scale in scales:

            acc_dict[scale] = list()
            ent_id2result_all[scale] = dict()
            rel_id2result_all[scale] = dict()
            ent_id2result_dict = dict()
            rel_id2result_dict = dict()
            for llm_name in llm_names:
                acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, acc_8, acc_9 = 0, 0, 0, 0, 0, 0, 0, 0, 0
                ent_id2result_list = list()
                rel_id2result_list = list()
                print('\nLLM %s' % llm_name)

                # Entity-Level
                print('\nEntity-Level')

                filename = f'../results/{kg_name}_{scale}/{kg_name}_entity_level_task_ET_{llm_name}.json'
                if os.path.exists(filename):
                    ent_id2result, rel_id2result, acc_1 = evaluate_mcq(filename)
                    ent_id2result_list.append(ent_id2result)
                    rel_id2result_list.append(rel_id2result)

                filename = f'../results/{kg_name}_{scale}/{kg_name}_entity_level_task_EC_{llm_name}.json'
                if os.path.exists(filename):
                    ent_id2result, rel_id2result, acc_2 = evaluate_mcq(filename)
                    ent_id2result_list.append(ent_id2result)
                    rel_id2result_list.append(rel_id2result)

                filename = f'../results/{kg_name}_{scale}/{kg_name}_entity_level_task_ED_{llm_name}.json'
                if os.path.exists(filename):
                    ent_id2result, rel_id2result, acc_3 = evaluate_cls(filename)
                    ent_id2result_list.append(ent_id2result)
                    rel_id2result_list.append(rel_id2result)

                # Relation-Level
                print('\nRelation-Level')

                filename = f'../results/{kg_name}_{scale}/{kg_name}_relation_level_task_RT_{llm_name}.json'
                if os.path.exists(filename):
                    ent_id2result, rel_id2result, acc_4 = evaluate_mcq(filename)
                    ent_id2result_list.append(ent_id2result)
                    rel_id2result_list.append(rel_id2result)

                filename = f'../results/{kg_name}_{scale}/{kg_name}_relation_level_task_FC_{llm_name}.json'
                if os.path.exists(filename):
                    ent_id2result, rel_id2result, acc_5 = evaluate_cls(filename)
                    ent_id2result_list.append(ent_id2result)
                    rel_id2result_list.append(rel_id2result)

                filename = f'../results/{kg_name}_{scale}/{kg_name}_relation_level_task_RP_{llm_name}.json'
                if os.path.exists(filename):
                    ent_id2result, rel_id2result, acc_6 = evaluate_mcq(filename)
                    ent_id2result_list.append(ent_id2result)
                    rel_id2result_list.append(rel_id2result)

                # Subgraph-Level
                print('\nSubgraph-Level')

                filename = f'../results/{kg_name}_{scale}/{kg_name}_subgraph_level_task_ER_{llm_name}.json'
                if os.path.exists(filename):
                    ent_id2result, rel_id2result, acc_7 = evaluate_mcq(filename)
                    ent_id2result_list.append(ent_id2result)
                    rel_id2result_list.append(rel_id2result)

                filename = f'../results/{kg_name}_{scale}/{kg_name}_subgraph_level_task_R1_{llm_name}.json'
                if os.path.exists(filename):
                    ent_id2result, rel_id2result, acc_8 = evaluate_cls(filename)
                    ent_id2result_list.append(ent_id2result)
                    rel_id2result_list.append(rel_id2result)

                filename = f'../results/{kg_name}_{scale}/{kg_name}_subgraph_level_task_R2_{llm_name}.json'
                if os.path.exists(filename):
                    ent_id2result, rel_id2result, acc_9 = evaluate_mcq(filename)
                    ent_id2result_list.append(ent_id2result)
                    rel_id2result_list.append(rel_id2result)

                acc_dict[scale].append([
                    acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, acc_8, acc_9
                ])

                ent_id2result_dict[llm_name] = ent_id2result_list
                rel_id2result_dict[llm_name] = rel_id2result_list

            ent_id2result_all[scale] = ent_id2result_dict
            rel_id2result_all[scale] = rel_id2result_dict

        print('')
        print(json.dumps(acc_dict, indent=4))

        # Knowledge-Level Evaluation

        # get Coverage metrics
        for scale in scales:

            kg = construct_kg(
                path_rel_triple='../kg_data/%s/%s/%s_%s_relations.csv' % (kg_name, scale, kg_name, scale),
                path_attr_triple='../kg_data//%s/%s/%s_%s_attributes.csv' % (kg_name, scale, kg_name, scale),
                print_stats=False
            )
            for llm_name in llm_names:
                print('\nLLM %s\t %s %s' % (llm_name, kg_name, scale))
                print('Entity-Level (AVG)')
                compute_coverage_ratio_avg(ent_id2result_all[scale][llm_name])
                print('Entity-Level (Deg-Aware)')
                compute_coverage_ratio_degree_aware(ent_id2result_all[scale][llm_name], kg, aspect='ent')
                print('Relation-Level (AVG)')
                compute_coverage_ratio_avg(rel_id2result_all[scale][llm_name])
                print('Relation-Level (Deg-Aware)')
                compute_coverage_ratio_degree_aware(rel_id2result_all[scale][llm_name], kg, aspect='rel')
                print('Triple-Level (AVG)')
                compute_coverage_triple(ent_id2result_all[scale][llm_name], rel_id2result_all[scale][llm_name], kg)
