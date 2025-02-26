# -*- coding: utf-8 -*-
import os
import random
import numpy as np
from tqdm import tqdm


"""
KnowledgeGraph triple: [head@@type, relation, tail@@type]
"""


random.seed(42)
# offset for entity ID
OFFSET = 20000


class Entity:
    """
    Entity object in KnowledgeGraph
    --
    param idx
        unique ID for each entity
    param str
        name string for each entity
    param is_literal
        True for attribute node & False for entity node
    param affiliation
        affiliated KnowledgeGraph object
    """
    def __init__(self, idx: int, name: str, etype: str,
                 is_literal=False, affiliation=None):
        self._is_literal = is_literal
        self.id = idx
        self.name = name.strip()
        self.etype = etype.strip()
        self.value = name.strip()
        self.affiliation = affiliation

        self.frequency = 0
        self.involved_as_head_dict = dict()
        self.involved_as_tail_dict = dict()
        # entity embedding
        self.embedding = None

    @staticmethod
    def is_entity():
        return True

    @staticmethod
    def is_relation():
        return False

    def is_literal(self):
        return self._is_literal

    def add_relation_as_head(self, relation, tail):
        if self.involved_as_head_dict.__contains__(relation) is False:
            self.involved_as_head_dict[relation] = set()
        self.involved_as_head_dict[relation].add(tail)
        self.frequency += 1

    def add_relation_as_tail(self, relation, head):
        if self.involved_as_tail_dict.__contains__(relation) is False:
            self.involved_as_tail_dict[relation] = set()
        self.involved_as_tail_dict[relation].add(head)
        self.frequency += 1

    def check_triple_with_entity(self, ent):
        for key, val_set in self.involved_as_head_dict.items():
            if ent in val_set:
                return [self, key, ent]
        for key, val_set in self.involved_as_tail_dict.items():
            if ent in val_set:
                return [ent, key, self]
        return None


class Relation:
    """
    Relation object in KnowledgeGraph
    --
    param idx
        unique ID for each relation
    param str
        name string for each relation
    param is_attribute
        True for attribute edge & False for relation edge
    param affiliation
        affiliated KnowledgeGraph object
    """
    def __init__(self, idx: int, name: str,
                 is_attribute=False, affiliation=None):
        self._is_attribute = is_attribute
        self.id = idx
        self.name = name.strip()
        self.value = name.strip()
        self.affiliation = affiliation

        self.frequency = 0
        self.head_ent_set = set()
        self.tail_ent_set = set()
        self.triple_set = set()
        # (inverse) functionality of relation
        self.functionality = 0.0
        self.functionality_inv = 0.0
        # relation embedding
        self.embedding = None

    @staticmethod
    def is_entity():
        return False

    @staticmethod
    def is_relation():
        return True

    def is_attribute(self):
        return self._is_attribute

    def add_relation_triple(self, head, tail):
        self.head_ent_set.add(head)
        self.tail_ent_set.add(tail)
        self.triple_set.add((head, tail))
        self.frequency += 1

    def calculate_functionality(self):
        """ calculate both functionality & inverse functionality
        """
        if self.frequency == 0:
            return
        self.functionality = len(self.head_ent_set) / self.frequency
        self.functionality_inv = len(self.tail_ent_set) / self.frequency


class KnowledgeGraph:
    """
    Knowledge Graph object
    --
    param name
        name of KnowledgeGraph
    param inverse_triple
        whether or not to involve inverse triples
    """
    def __init__(self, name=None, inverse_triple=False):
        self.name = name
        self.inverse_triple = inverse_triple

        # set variables for ents/rels/attrs/literals
        self.ents_set = set()
        self.rels_set = set()
        self.attrs_set = set()
        self.lites_set = set()

        # dict variables for ents/rels/attrs/literals (by name)
        self.ents_dict_by_name = dict()
        self.rels_dict_by_name = dict()
        self.attrs_dict_by_name = dict()
        self.lites_dict_by_name = dict()

        # dict variables for ents/rels/attrs/literals (by ID)
        self.ents_dict_by_id = dict()
        self.rels_dict_by_id = dict()

        # triple lists for rels/attrs
        self.rels_triple_list = list()
        self.attrs_triple_list = list()

        # list of entity IDs
        self.ent_id_list = list()
        # True if it is literal & False if it is entity
        self.is_literal_list = list()

        # functionality variables for rels/attrs
        self.rels_set_func_ranked = list()
        self.rels_set_func_inv_ranked = list()
        self.attrs_set_func_ranked = list()
        self.attrs_set_func_inv_ranked = list()

        # functionality helper variables
        self.functionality_dict = dict()
        self.fact_dict_by_head = dict()
        self.fact_dict_by_tail = dict()

        # pinyin inv dict
        self.pinyin_dict = dict()

        # entity embedding
        self.ent_embeddings = None

        # for KG embedding (name not object)
        self.triple_num = 0
        self.triples_all = list()
        self.triples_train = list()
        self.triples_valid = list()
        self.triples_test = list()
        # KG entity list (only names)
        self.ents_list = list()
        # KG relation list (only names)
        self.rels_list = list()
        # name2id mapping
        self.ent2id_dict = dict()
        self.id2ent_dict = dict()
        self.rel2id_dict = dict()
        self.id2rel_dict = dict()
        # get triple pools
        self.triple_pool_train = set()
        self.triple_pool_golden = set()
        # for KG embedding
        self.n_ents = 0
        self.n_rels = 0
        self.n_triples_train = 0
        self.n_triples_valid = 0
        self.n_triples_test = 0

    def get_entity(self, name):
        """
        get Entity object with given name
        """
        if self.ents_dict_by_name.__contains__(name):
            return self.ents_dict_by_name.get(name)
        ent_name, ent_type = name.split('@@')
        entity = Entity(
            idx=len(self.lites_set) + len(self.ents_set),
            name=ent_name, etype=ent_type, affiliation=self
        )
        self.ents_set.add(entity)
        self.ents_dict_by_name[name] = entity
        self.ents_dict_by_id[entity.id] = entity
        self.ent_id_list.append(entity.id)
        self.is_literal_list.append(False)
        return entity

    def get_relation(self, name):
        """
        get Relation object with given name
        """
        if self.rels_dict_by_name.__contains__(name):
            return self.rels_dict_by_name.get(name)
        relation = Relation(
            idx=len(self.attrs_set) + len(self.rels_set),
            name=name, affiliation=self
        )
        self.rels_set.add(relation)
        self.rels_dict_by_name[relation.name] = relation
        self.rels_dict_by_id[relation.id] = relation
        return relation

    def get_attribute(self, name):
        """
        get Attribute(Relation) object with given name
        """
        if self.attrs_dict_by_name.__contains__(name):
            return self.attrs_dict_by_name.get(name)
        attribute = Relation(
            idx=len(self.attrs_set) + len(self.rels_set),
            name=name, affiliation=self, is_attribute=True
        )
        self.attrs_set.add(attribute)
        self.attrs_dict_by_name[attribute.name] = attribute
        return attribute

    def get_literal(self, name):
        """
        get Literal(Entity) object with given
        """
        if self.lites_dict_by_name.__contains__(name):
            return self.lites_dict_by_name.get(name)
        literal = Entity(
            idx=len(self.lites_set) + len(self.ents_set),
            name=name, affiliation=self, is_literal=True
        )
        self.lites_set.add(literal)
        self.lites_dict_by_name[literal.name] = literal
        self.is_literal_list.append(True)
        return literal

    def insert_relation_triple(self, head, relation, tail):
        """
        insert one given relation triple
        """
        ent_h = self.get_entity(head)
        rel = self.get_relation(relation)
        ent_t = self.get_entity(tail)
        self.__insert_relation_triple_one_way(ent_h, rel, ent_t)
        if self.inverse_triple:
            relation_inv = relation.strip() + str('-(INV)')
            rel_inv = self.get_relation(relation_inv)
            self.__insert_relation_triple_one_way(ent_t, rel_inv, ent_h)

    def insert_attribute_triple(self, entity, attribute, literal):
        """
        insert one given attribute triple
        """
        ent = self.get_entity(entity)
        attr = self.get_attribute(attribute)
        lite = self.get_literal(literal)
        self.__insert_attribute_triple_one_way(ent, attr, lite)
        if self.inverse_triple:
            attribute_inv = attribute.strip() + str('-(INV)')
            attr_inv = self.get_attribute(attribute_inv)
            self.__insert_attribute_triple_one_way(lite, attr_inv, ent)

    def __insert_relation_triple_one_way(self, ent_h, rel, ent_t):
        ent_h.add_relation_as_head(relation=rel, tail=ent_t)
        rel.add_relation_triple(head=ent_h, tail=ent_t)
        ent_t.add_relation_as_tail(relation=rel, head=ent_h)
        self.rels_triple_list.append((ent_h, rel, ent_t))
        if not self.fact_dict_by_head.__contains__(ent_h.id):
            self.fact_dict_by_head[ent_h.id] = list()
        if not self.fact_dict_by_tail.__contains__(ent_t.id):
            self.fact_dict_by_tail[ent_t.id] = list()
        self.fact_dict_by_head[ent_h.id].append((rel.id, ent_t.id))
        self.fact_dict_by_tail[ent_t.id].append((rel.id, ent_h.id))

    def __insert_attribute_triple_one_way(self, ent, attr, lite):
        ent.add_relation_as_head(relation=attr, tail=lite)
        attr.add_relation_triple(head=ent, tail=lite)
        lite.add_relation_as_tail(relation=attr, head=ent)
        self.attrs_triple_list.append((ent, attr, lite))
        if not self.fact_dict_by_head.__contains__(ent.id):
            self.fact_dict_by_head[ent.id] = list()
        if not self.fact_dict_by_tail.__contains__(lite.id):
            self.fact_dict_by_tail[lite.id] = list()
        self.fact_dict_by_head[ent.id].append((attr.id, lite.id))
        self.fact_dict_by_tail[lite.id].append((attr.id, ent.id))

    def get_object_by_name(self, name):
        """
        get Entity/Relation/KG object by the name (external access)
        """
        name = name.strip()
        if self.ents_dict_by_name.__contains__(name):
            return self.ents_dict_by_name[name]
        if self.rels_dict_by_name.__contains__(name):
            return self.rels_dict_by_name[name]
        if self.attrs_dict_by_name.__contains__(name):
            return self.attrs_dict_by_name[name]
        if self.lites_dict_by_name.__contains__(name):
            return self.lites_dict_by_name[name]

    def calculate_functionality(self):
        """
        calculate relation functionality (https://arxiv.org/pdf/2105.05596.pdf)
        """
        for relation in self.rels_set:
            relation.calculate_functionality()
            self.rels_set_func_ranked.append(relation)
            self.rels_set_func_inv_ranked.append(relation)
            self.functionality_dict[relation.id] = relation.functionality
        for attribute in self.attrs_set:
            attribute.calculate_functionality()
            self.attrs_set_func_ranked.append(attribute)
            self.attrs_set_func_inv_ranked.append(attribute)
            self.functionality_dict[attribute.id] = attribute.functionality
        self.rels_set_func_ranked.sort(
            key=lambda x: x.functionality, reverse=True
        )
        self.rels_set_func_inv_ranked.sort(
            key=lambda x: x.functionality_inv, reverse=True
        )
        self.attrs_set_func_ranked.sort(
            key=lambda x: x.functionality, reverse=True
        )
        self.attrs_set_func_inv_ranked.sort(
            key=lambda x: x.functionality_inv, reverse=True
        )

    def calculate_pinyin(self, func):
        """
        calculate entity pinyin
        """
        for ent in self.ents_set:
            p = func(ent.name)
            if not self.pinyin_dict.__contains__(p):
                self.pinyin_dict[p] = list()
            self.pinyin_dict[p].append(ent)

    def init_ent_embeddings(self):
        """
        initialize entity embeddings
        """
        for ent in self.ents_set:
            idx, embedding = ent.id, ent.embedding
            if embedding is None:
                break
            if self.ent_embeddings is None:
                self.ent_embeddings = np.zeros((
                    len(self.ents_set), len(embedding)
                ))
            self.ent_embeddings[idx, :] = embedding

    def set_ent_embedding(self, idx, emb, func=None):
        """
        set entity embeddings
        """
        if self.ent_embeddings is not None:
            if func is None:
                self.ent_embeddings[idx, :] = emb
            else:
                self.ent_embeddings[idx, :] = func(
                    self.ents_dict_by_id[idx].embedidng, emb
                )

    def prepare_kg_triples(self, train_ratio=[0.7, 0.9, 1.0]):
        """
        prepare KG triples for model training (train/valid/test)
        """
        # get list of ent/rel names
        self.ents_list = list(self.ents_dict_by_name.keys())
        self.rels_list = list(self.rels_dict_by_name.keys())
        for ent in self.ents_list:
            self.ent2id_dict[ent] = len(self.ent2id_dict) + OFFSET
            self.id2ent_dict[self.ent2id_dict[ent]] = ent
        for rel in self.rels_list:
            self.rel2id_dict[rel] = len(self.rel2id_dict)
            self.id2rel_dict[self.rel2id_dict[rel]] = rel
        for triple in self.rels_triple_list:
            head = triple[0].name
            rel = triple[1].name
            tail = triple[2].name
            self.triples_all.append((
                self.ent2id_dict[head],
                self.rel2id_dict[rel],
                self.ent2id_dict[tail]
            ))
        # train/valid/test split
        self.triples_num = len(self.triples_all)
        print('triple number: %d' % self.triples_num)
        random.shuffle(self.triples_all)
        self.triples_train = self.triples_all[
            :int(self.triples_num * train_ratio[0])
        ]
        self.triples_valid = self.triples_all[
            int(self.triples_num * train_ratio[0]):
            int(self.triples_num * train_ratio[1])
        ]
        self.triples_test = self.triples_all[
            int(self.triples_num * train_ratio[1]):
        ]
        self.triple_pool_train = set(self.triples_train)
        self.triple_pool_golden = set(self.triples_train) | set(self.triples_valid)
        self.n_ents = len(self.ents_set)
        self.n_rels = len(self.rels_set)
        self.n_triples_train = len(self.triples_train)
        self.n_triples_valid = len(self.triples_valid)
        self.n_triples_test = len(self.triples_test)

    def next_raw_batch(self, batch_size: int):
        idx_random = np.random.permutation(len(self.triples_train))
        start = 0
        while start < len(self.triples_train):
            end = min(start + batch_size, len(self.triples_train))
            yield [self.triples_train[ii] for ii in idx_random[start:end]]
            start = end

    def generate_train_batch(self, in_queue, out_queue):
        """ training generator
        """
        while True:
            raw_batch = in_queue.get()
            if raw_batch is None:
                return
            else:
                batch_pos = raw_batch
                batch_neg = []
                corrupt_head_prob = np.random.binomial(1, 0.5)
                for ent_h, rel, ent_t in batch_pos:
                    neg_h = ent_h
                    neg_t = ent_t
                    while True:
                        if corrupt_head_prob:
                            neg_h = self.ent2id_dict[
                                random.choice(list(self.ents_list))
                            ]
                        else:
                            neg_t = self.ent2id_dict[
                                random.choice(list(self.ents_list))
                            ]
                        if (neg_h, rel, neg_t) not in self.triple_pool_train:
                            break
                    batch_neg.append([neg_h, rel, neg_t])
                out_queue.put((batch_pos, batch_neg))

    def get_kg_statistics(self, output=True, output_func=False):
        """
        get and print KG statistics
        """
        stats = dict()
        stats['kg-name'] = self.name
        stats['ents-num'] = int(len(self.ents_set))
        stats['ent-types-num'] = len(set([ent.etype for ent in self.ents_set]))
        if self.inverse_triple:
            stats['rels-num'] = int(len(self.rels_set) / 2)
            stats['attrs-num'] = int(len(self.attrs_set) / 2)
            stats['rels-triple'] = int(len(self.rels_triple_list) / 2)
            stats['attrs-triple'] = int(len(self.attrs_triple_list) / 2)
        else:
            stats['rels-num'] = int(len(self.rels_set))
            stats['attrs-num'] = int(len(self.attrs_set))
            stats['rels-triple'] = int(len(self.rels_triple_list))
            stats['attrs-triple'] = int(len(self.attrs_triple_list))
        stats['lites-num'] = int(len(self.lites_set))
        # print KG statistics
        if output:
            print('Knowledge Graph Statistics')
            print('KG name: %s' % self.name)
            print('- entity number: %d' % stats['ents-num'])
            print('- entity type number: %d' % stats['ent-types-num'])
            print('- relation number: %d' % stats['rels-num'])
            print('- attribute number: %d' % stats['attrs-num'])
            print('- literal number: %d' % stats['lites-num'])
            print('- relation triple number: %d' % stats['rels-triple'])
            print('- attribute triple number: %d' % stats['attrs-triple'])
        # Count TOP-100 frequent entities
        stats['top-freq-ents'] = []
        for ent in self.ents_set:
            stats['top-freq-ents'].append((ent.name, ent.frequency))
        stats['top-freq-ents'] = sorted(
            stats['top-freq-ents'], key=lambda x: x[1], reverse=True
        )[:100]
        # if output:
        #     print('Top-100 frequent entities:')
        #     for ent in stats['top-freq-ents']:
        #         print('%s\t%d' % (ent[0], ent[1]))
        # Count TOP-10 frequent relations
        stats['top-freq-rels'] = []
        for rel in self.rels_set:
            stats['top-freq-rels'].append((rel.name, rel.frequency))
        stats['top-freq-rels'] = sorted(
            stats['top-freq-rels'], key=lambda x: x[1], reverse=True
        )[:10]
        # if output:
        #     print('Top-10 frequent relations:')
        #     for rel in stats['top-freq-rels']:
        #         print('%s\t%d' % (rel[0], rel[1]))
        return stats


def construct_kg(
    path_rel_triple, path_attr_triple,
    sep='\t', name='', inverse_triple=False,
    output_stats=True, output_func=False,
    entity_id=False, h_t_r=False,
    print_stats=True
):
    """
    construct a KG object with given rel/attr triple files
    --
    param header
        whether or not rel/attr triple files have header
    param index
        whether or not rel/attr triple files have entity ID
    param h_t_r
        whehter or not rel triple files in head-tail-rel format
    """
    kg = KnowledgeGraph(name=name, inverse_triple=inverse_triple)
    # load relation triples
    with open(path_rel_triple, 'r', encoding='utf-8') as f_in:
        for line in tqdm(f_in.readlines(), desc='building KG relation triples'):
            if len(line.strip()) == 0 or 'ID' in line.strip():
                continue
            params = line.strip().split(sep=sep)
            if len(params) != 3 and not entity_id:
                print(line)
                continue
            if not entity_id and not h_t_r:
                hh = params[0].strip()
                rr = params[1].strip()
                tt = params[2].strip()
            elif entity_id and h_t_r:
                hh = params[1].strip()
                rr = params[4].strip()
                tt = params[3].strip()
                hh = params[1].strip()
            kg.insert_relation_triple(hh, rr, tt)
    # load attribute triples
    if os.path.isfile(path_attr_triple):
        with open(path_attr_triple, 'r', encoding='utf-8') as f_in:
            for line in tqdm(f_in.readlines(), desc='building KG attribute triples'):
                if len(line.strip()) == 0 or 'ID' in line.strip():
                    continue
                params = line.strip().split(sep=sep)
                if len(params) != 3 and not entity_id:
                    print(line)
                    continue
                if not entity_id:
                    ee = params[0].strip()
                    aa = params[1].strip()
                    ll = params[2].strip()
                    ee = params[0].strip()
                    aa = params[1].strip()
                    ll = params[2].strip()
                else:
                    ee = params[1].strip()
                    aa = params[2].strip()
                    ll = params[3].strip()
                kg.insert_attribute_triple(ee, aa, ll)

    # get KG statistics
    if print_stats:
        kg.get_kg_statistics(output=output_stats, output_func=output_func)

    return kg
