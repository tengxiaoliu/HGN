# [1103] ver 14: add new build graph method
import re
import torch
import torch.nn.functional as F
import graphviz
import spacy
import numpy as np
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import dgl
from transformers import BertModel, BertConfig, BertTokenizer, AutoTokenizer
from IPython.display import Image

import logging

logging.getLogger('allennlp.common.params').disabled = True
logging.getLogger('allennlp.nn.initializers').disabled = True
logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)
logging.getLogger('urllib3.connectionpool').disabled = True

nlp = spacy.load("en_core_web_sm")
predictorSRL = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
predictorCoref = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")
# BERTtokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
BERTtokenizer = AutoTokenizer.from_pretrained('roberta-large')
# BERTmodel = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
# Casedtokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

MAX_LEN = 512


class Node(object):
    def __init__(self, text, tag, embedding=[0], word_pos=[-1, -1], char_pos=[-1, -1], token_pos=[-1, -1]):
        self.text = text
        self.tag = tag
        self.word_pos = word_pos  # 左闭右开区间
        self.char_pos = char_pos  # 闭区间
        self.embedding = embedding
        self.token_pos = token_pos

    def __str__(self):
        return '{0:} {1:} {2:}-{3:} {4:}-{5:} {6:}-{7:}'.format(self.text, self.tag, self.word_pos[0], self.word_pos[1],
                                                                self.char_pos[0], self.char_pos[1], self.token_pos[0],
                                                                self.token_pos[1])

    def __repr__(self):
        return self.__str__()

    def get_token_pos(self):
        return self.token_pos

    def get_embedding(self):
        return self.embedding


class EntityNode(Node):
    def __init__(self, text, tag, embedding=[0], word_pos=[-1, -1], char_pos=[-1, -1], token_pos=[-1, -1], ancestor=-1):
        self.text = text
        self.tag = tag
        self.word_pos = word_pos  # 左闭右开区间
        self.char_pos = char_pos  # 闭区间
        self.embedding = embedding
        self.token_pos = token_pos
        self.ancestor = ancestor  # 祖先sentence node的id

    def __str__(self):
        return '{0:} {1:} {2:}-{3:} {4:}-{5:} {6:}-{7:} Sent:{8:}'.format(self.text, self.tag,
                                                                          self.word_pos[0], self.word_pos[1],
                                                                          self.char_pos[0], self.char_pos[1],
                                                                          self.token_pos[0], self.token_pos[1],
                                                                          self.ancestor)

    def __repr__(self):
        return self.__str__()


def do_SRL(doc):
    predSRL = predictorSRL.predict(
        sentence=doc
    )
    return predSRL


def do_Coref(doc):
    predCoref = predictorCoref.predict(
        document=doc
    )
    return predCoref


def get_char_pos(words, st, ed):
    str1 = ''
    char_st = -1
    char_ed = -1
    for i, w in enumerate(words):
        if i >= st and char_st == -1:
            char_st = len(str1)
        if i >= ed and char_ed == -1:
            char_ed = len(str1) - 1  # pos index是一个闭区间
        if w != '\xa0' and w != ' ':
            str1 = str1 + w.replace(' ', '')
    if char_ed == -1:
        char_ed = len(str1)

    return char_st, char_ed


def find_node_by_char_pos(nodes, char_pos):
    ret = []
    st, ed = char_pos
    for i, n in enumerate(nodes):
        if n.char_pos[0] <= st and n.char_pos[1] >= ed:
            ret.append(i)
    return ret


def find_node_by_word_pos(nodes, word_pos):
    ret = []
    st, ed = word_pos
    for i, n in enumerate(nodes):
        if n.pos[0] <= st and n.pos[1] >= ed:
            ret.append(i)
    return ret


# combine verb items
def parse_SRL_item_v5(doc_words, sent_words, verb_item, global_idx, article_encoded, sent_id):
    """
    article_encoded: should be tokenized passage
    return: arglist [arg type, arg text, word position, char position]
    ver 14: add sentence nodes
    """
    tags = verb_item['tags']

    b_idx = -1
    e_idx = -1
    pos_list = []
    for i, t in enumerate(tags):
        if i > 0 and (t[0] == 'I' or t[0] == 'E') and (tags[i - 1][0] == 'B' or tags[i - 1][0] == 'I') and (
                tags[i - 1][1:] == t[1:]):
            pass
        elif i > 0 and tags[i - 1][0] != 'O':
            # char_b, char_e = get_char_pos(sent_words, b_idx, i)
            global_char_b, global_char_e = get_char_pos(doc_words, global_idx + b_idx, global_idx + i)

            # token_pos in node should be the indices of tokenized input ids (closed), not ids themselves
            # token ids are a list of token ids, local pos should take [ids[0], ids[-1]]
            # Given encoded whole passage, we only need the global token ids for current node text
            global_token_ids = get_arg_token_ids(article_encoded, global_char_b, global_char_e)
            assert global_token_ids.sum() == global_token_ids.sum(), "NaN in input ids"

            if len(global_token_ids) == 0:
                print("global_token_ids len 0!")
                print("Words:", sent_words[b_idx: i])
                print("Word idx:", global_idx + b_idx, global_idx + i)
                print("Sentence:", sent_words)
                print("Article:", doc_words)
                print("==================PASSED ONE ITEM...====================")
                continue

            pos_list.append([
                tags[b_idx][2:],
                sent_words[b_idx: i],
                [global_idx + b_idx, global_idx + i],  # global word pos
                [global_char_b, global_char_e],  # global char pos
                [global_token_ids[0], global_token_ids[-1]],
                # global token ids, stored in token_pos in Node object, 闭区间
                sent_id  # sentence id, ancestor of current entity node
            ])

            # if i == 20:
            #     print("Parse srl item v3")
            #     print("sent:", sent_words)
            #     print("words:", sent_words[b_idx: i])
            #     print("ids:", global_token_ids)
            #     print("tokens:", BERTtokenizer.convert_ids_to_tokens(encoded.input_ids[global_token_ids[0]: global_token_ids[-1]+1]))

            b_idx = i
        else:
            b_idx = i

    if b_idx < len(tags) and tags[-1][0] != 'O':
        # word_type.append(tags[-1][2:])
        # char_b, char_e = get_char_pos(sent_words, b_idx, i)
        global_char_b, global_char_e = get_char_pos(doc_words, global_idx + b_idx, global_idx + i)
        # char_e += len(sent_words[-1])
        global_char_e += len(sent_words[-1])
        # input_ids = encoded.input_ids

        # token_pos in node should be the indices of tokenized input ids (closed), not ids themselves
        # token ids are a list of toke ids, local pos should take [ids[0], ids[-1]]
        global_token_ids = get_arg_token_ids(article_encoded, global_char_b, global_char_e)
        assert global_token_ids.sum() == global_token_ids.sum(), "NaN in input ids"

        pos_list.append([
            tags[b_idx][2:],
            sent_words[b_idx: i],
            [global_idx + b_idx, global_idx + i],  # global pos
            [global_char_b, global_char_e],  # global char pos
            [global_token_ids[0], global_token_ids[-1]],  # global token ids, stored in token_pos in Node object, 闭区间
            sent_id  # sentence id, ancestor of current entity node
        ])

    return pos_list


# parse document by sentence

def parse_doc_by_sent_v5(sentences):
    """
    For BERT finetuning
    returns a list of sent-arg list
    for ver 14: add sentence nodes
    for hotpotqa: remove tokenizer, only keep the token text
    """
    # doc = nlp(text)
    global_idx = 0
    global_char_sum = 0
    doc_arg_list = []
    doc_words = []
    sent_nodes_list = []
    # encode the whole passage
    text = ' '.join(sentences)

    article_encoded = get_sent_encoded(text)
    for sent_id, sent in enumerate(sentences):
        sent_arg_list = []
        sent_res = do_SRL(sent)
        # encoded, sent_emb = get_sent_embedding(sent.text)
        # encoded = get_sent_encoded(sent.text)
        sent_words = sent_res['words']
        doc_words.extend(sent_words)

        if global_idx > 1200:
            print("Long article > 1200.")

        for verb_item in sent_res['verbs']:
            sent_arg_list.append(
                parse_SRL_item_v5(doc_words, sent_words, verb_item, global_idx, article_encoded, sent_id))
        doc_arg_list.extend(sent_arg_list)

        # ver14: get global token ids in article_encoded, for a sentence
        this_char_num = len(''.join(sent_words))

        # global_char_b, global_char_e = get_char_pos(doc_words, global_idx, global_idx+len(sent_words))
        global_token_ids = get_arg_token_ids(article_encoded, global_char_sum, global_char_sum + this_char_num)

        sent_nodes_list.append(
            Node(text=sent_words,
                 tag='sent',
                 word_pos=[global_idx, global_idx + len(sent_words)],
                 char_pos=[global_char_sum, global_char_sum + this_char_num],
                 token_pos=[global_token_ids[0], global_token_ids[-1]]  # global token ids, stored in token_pos
                 )
        )

        global_idx += len(sent_words)  # word index
        global_char_sum += this_char_num  # char index
    return doc_arg_list, article_encoded, sent_nodes_list, global_idx, global_char_sum


def build_graph_hotpot(sentences, question, ante=True):
    """
    For BERT fine-tuning
    hotpotQA: input is document and question
    document is a list of sentences
    """
    nodes = []
    edges = []
    document = ' '.join(sentences)

    # print(Coref)
    # if len(Coref['clusters']) == 0:
    #     print("No coreference.")
    #     return False, False, False

    verb_id = -1
    doc_arg_list, article_encoded, sent_nodes, global_idx, global_char_sum = parse_doc_by_sent_v5(sentences)
    for arg_list in doc_arg_list:
        # print(arg_list)
        st = len(nodes)
        for arg in arg_list:
            # char_pos = get_char_pos(words, arg[2], arg[3])
            nodes.append(
                EntityNode(text=arg[1],
                           tag=arg[0],
                           word_pos=arg[2],
                           char_pos=arg[3],
                           token_pos=arg[4],  # global token ids, stored in token_pos
                           ancestor=arg[5]
                           )
            )
            if arg[0] == 'V':
                verb_id = len(nodes) - 1
        for i in range(st, len(nodes)):
            if nodes[i].tag != 'V':
                edges.append([verb_id, nodes[i].tag, i])

    # print(nodes)
    # ver 15 TODO: this is where we need to take extra care with q+opt node, in the unified node form
    # add q_opt nodes, as sent nodes, but stored in entity nodes list, for the convenience of coreference
    # global id may not be accurate, only need to make sure char index is accurate
    global_encoded_sum = len(article_encoded['input_ids']) - 1

    qo_text = question
    qo_words = qo_text.split()
    qo_encoded = get_sent_encoded(qo_text)
    document += qo_text

    nodes.append(
        Node(text=qo_words,
             tag='query',
             word_pos=[global_idx, global_idx + len(qo_words)],
             char_pos=[global_char_sum, global_char_sum + len(''.join(qo_words))],
             token_pos=[global_encoded_sum, global_encoded_sum + len(qo_encoded['input_ids']) - 3],
             # global token ids, stored in token_pos TODO: how to encode q_opt?
             )
    )
    global_idx += len(qo_words)
    global_char_sum += len(''.join(qo_words))
    global_encoded_sum += len(qo_encoded['input_ids']) - 2

    # TODO: checked index, ut not edge coref yet

    num_edge = len(edges)
    num_node = len(nodes)

    Coref = do_Coref(document)
    doc = Coref['document']
    for clu in Coref['clusters']:
        ref_nodes = []
        for it in clu:
            pos = get_char_pos(doc, it[0], it[1] + 1)
            ref_nodes.extend(find_node_by_char_pos(nodes, pos))
        if ante:
            nodes_ante = ref_nodes[:1]
        else:
            nodes_ante = ref_nodes
        for i, node1 in enumerate(nodes_ante):
            for node2 in ref_nodes[i + 1:]:
                if num_node - node1 <= 4 and num_node - node2 <= 4:
                    continue
                edges.append([node1, 'Coref', node2])

    # if len(edges) - num_edge == 0:
    #     print("\nCoreference error.")
    #     return False, False, False
    # new_art_encoded = get_sent_encoded(document)

    return nodes, edges, article_encoded, sent_nodes


########### Add word embedding

def get_encoded_sent(sent, tokenizer=BERTtokenizer):
    """
    First tokenize a sentence, then recompose them to a space separated sentence
    """
    tokens = tokenizer.tokenize(sent, return_tensors="pt")
    word_list = []
    prev_tok = ''
    for i, token in enumerate(tokens):
        if i == 0:
            prev_tok = token
            continue
        if token[0] == '#':
            prev_tok += token[2:]
        else:
            word_list.append(prev_tok)
            prev_tok = token
    word_list.append(prev_tok)
    return " ".join(word_list)


def get_sent_encoded(sent, tokenizer=BERTtokenizer):
    """
    For fine-tuning BERT
    Get sentence input ids only by tokenizer encoding
    """
    return tokenizer(sent)


def get_arg_token_ids(encoded, char_st, char_ed):
    """
    For fine-tuning BERT
    Get sentence input ids only by tokenizer encoding
    """
    tokens = encoded.tokens()
    input_ids = encoded.input_ids

    token_ids = []
    tok_str = ''

    for i, token in enumerate(tokens):
        if token == '[CLS]' or token == '[SEP]':
            continue
        if token[0] == '#':
            token = token[2:]
        tok_str += token
        strlen = len(tok_str)
        # if token[0] == '#':
        #     token = token[2:]
        if (strlen - 1) >= char_st and (strlen - 1) <= char_ed:
            token_ids.append(i)
        elif len(token_ids) == 0 and (strlen - 1) >= char_st and (strlen - 1) >= char_ed:
            token_ids.append(i)
            break

    arg_token_ids = np.array(token_ids)  # token id (index) for target argument (given by char pos)

    assert arg_token_ids.sum() == arg_token_ids.sum(), "NaN in input ids"
    # print("arg token ids:", arg_token_ids)
    # print("tokens:", tokens[arg_token_ids[0]:arg_token_ids[-1]+1])
    if len(token_ids) == 0:
        print("Token ids = 0")
        print("Tokens:", tokens)
        print("Char idx:", char_st, char_ed)

    return arg_token_ids


def get_token_ids(encoded, char_st, char_ed):
    """
    Get token ids from encoded tokens
    ? DO NOT use
    """
    tokens = encoded.tokens()
    token_ids = []
    tok_str = ''
    strlen = 0
    for i, token in enumerate(tokens):
        strlen = len(tok_str)
        if token[0] == '#':
            token = token[2:]
        if token == '[CLS]' or token == '[SEP]':
            continue
        if strlen >= char_st and strlen < char_ed:
            token_ids.append(i)
        tok_str += token
    return token_ids


#######visualise
def wrap(str1):
    ret = []
    ws = str1
    str2 = ''
    for w in ws:
        if len(str2) > 20:
            ret.append(str2)
            str2 = ''
        str2 += w + ' '
    ret.append(str2)
    return '\n'.join(ret)


def visualize(nodes, edges, img='graph'):
    g = graphviz.Digraph()
    # g.graph_attr['concentrate']='true'
    mentioned_x = set([x[0] for x in edges] + [x[2] for x in edges])

    for i, n in enumerate(nodes):
        if i in mentioned_x:
            g.node(str(i), str(n.tag + ':\n' + wrap(n.text)))
    for e in edges:
        x1, r, x2 = e
        if r != 'Coref':
            g.edge(str(x1), str(x2), label=r, color='red')
        else:
            g.edge(str(x1), str(x2), color='blue', dir='both')
    return Image(filename=g.render(img, format='png'))


############build dgl graph


def build_dgl_graph_v15(nodes, edges, sent_nodes):
    """
    Build DGL homogeneous graph based on New Graph structure.
    1112: checked graph through visualisation
    1116: qo_node feature: [-10, opt id]
    1201: hotpotQA, store node separately and combine
    """
    # split entity nodes and q_opt nodes
    entity_nodes = nodes[:-1]
    qo_nodes = nodes[-1:]
    last_e_node_id = len(entity_nodes) - 1

    # From top to end
    global_node_idx = 1
    global_node_feat = [torch.tensor([-1, -1])]
    global_src = []
    global_dst = []
    '''
    # root node as sent node 0
    sent_node_id = 0
    sent_node_feat = []
    r_s_edges = []  # feat: -2
    s_s_edges = []  # feat: -3
    '''
    # Step 1: root node -> ed_r_s -> sent node
    #         sent node -> ed_s_s -> sent node
    # Add unidirectional edge, then dgl.add_reverse_edges()
    global_node_idx += len(sent_nodes)
    tmp_ed_feat = []

    for s_idx, s_node in enumerate(sent_nodes):
        global_src.extend([0, global_node_idx])
        global_dst.extend([global_node_idx, s_idx + 1])

        # update node
        global_node_idx += 1
        tmp_ed_feat.append(torch.tensor([-2, -2]))  # root-sent node
        global_node_feat.append(torch.tensor(s_node.token_pos))  # sent node feat

        # add inter_sent node
        if s_idx < len(sent_nodes) - 1:
            global_src.extend([s_idx + 1, global_node_idx])
            global_dst.extend([global_node_idx, s_idx + 2])

            # update node
            global_node_idx += 1
            tmp_ed_feat.append(torch.tensor([-3, -3]))  # inter-sent node

    global_node_feat.extend(tmp_ed_feat)
    assert len(global_node_feat) == global_node_idx
    # PASSED test 1

    # Step 2: sent node -> ed_s_e -> entity node
    entity_node_base = global_node_idx
    global_node_idx += len(entity_nodes)
    tmp_ed_feat = []

    for e_idx, e_node in enumerate(entity_nodes):
        sent_node_id = e_node.ancestor + 1
        global_src.extend([sent_node_id, global_node_idx])
        global_dst.extend([global_node_idx, entity_node_base + e_idx])

        # update node
        global_node_idx += 1
        tmp_ed_feat.append(torch.tensor([-4, -4]))  # sent-entity node
        global_node_feat.append(torch.tensor(e_node.token_pos))  # sent node feat

    global_node_feat.extend(tmp_ed_feat)
    assert len(global_node_feat) == global_node_idx

    # Step 2.5: qo node -> ed_r_q -> root node
    # Stores qo node global node idx into qo_idx_lst
    qo_idx_lst = []
    for qo_id, q_node in enumerate(qo_nodes):
        global_src.extend([0, global_node_idx])
        global_dst.extend([global_node_idx, global_node_idx + 1])

        # update node
        global_node_feat.append(torch.tensor([-8, -8]))  # root-qo node
        global_node_idx += 1
        qo_idx_lst.append(global_node_idx)
        # global_node_feat.append(torch.tensor(q_node.token_pos))  # q_opt node feat
        # qo node feat change to [-10, qo_node_idx], for the convenience of encoding
        global_node_feat.append(torch.tensor([-10, qo_id]))
        global_node_idx += 1

    g = dgl.graph((torch.tensor(global_src), torch.tensor(global_dst)))
    g = dgl.add_reverse_edges(g)

    # PASSED test 2

    # Step 3: verb node -> ed_v_a -> arg node
    #         arg node -> ed_a_v -> verb node
    #         arg node <-> ed_a_a <-> arg node
    add_src = []
    add_dst = []
    for edge in edges:
        if edge[1] == 'Coref':
            if edge[2] > last_e_node_id:
                # sent-q edge: ed_s_q
                assert edge[0] <= last_e_node_id
                tmp_sent_node_id = entity_nodes[edge[0]].ancestor + 1
                tmp_qo_node_id = qo_idx_lst[edge[2] - last_e_node_id - 1]
                add_src.extend([tmp_sent_node_id, global_node_idx, tmp_qo_node_id, global_node_idx])
                add_dst.extend([global_node_idx, tmp_qo_node_id, global_node_idx, tmp_sent_node_id])
                # update node
                global_node_idx += 1
                global_node_feat.append(torch.tensor([-9, -9]))  # sent-q node
            else:
                # normal coref edge: ed_a_a
                add_src.extend([entity_node_base + edge[0], global_node_idx, entity_node_base + edge[2], global_node_idx])
                add_dst.extend([global_node_idx, entity_node_base + edge[2], global_node_idx, entity_node_base + edge[0]])
                # update node
                global_node_idx += 1
                global_node_feat.append(torch.tensor([-7, -7]))  # arg-arg node

        else:
            # srl edge: ed_v_a
            add_src.extend([entity_node_base + edge[0], global_node_idx])
            add_dst.extend([global_node_idx, entity_node_base + edge[2]])
            # update node
            global_node_idx += 1
            global_node_feat.append(torch.tensor([-5, -5]))  # verb-arg node

            # anti-srl edge: ed_a_v
            add_src.extend([entity_node_base + edge[2], global_node_idx])
            add_dst.extend([global_node_idx, entity_node_base + edge[0]])
            # update node
            global_node_idx += 1
            global_node_feat.append(torch.tensor([-6, -6]))  # arg-verb node

    g = dgl.add_edges(g, torch.tensor(add_src), torch.tensor(add_dst))
    assert len(global_node_feat) == g.num_nodes()

    global_node_feat = torch.stack(global_node_feat)
    g.ndata['pos'] = global_node_feat

    return g, len(sent_nodes)


def pad_seq(seq, max_len=MAX_LEN):
    """
    For BERT fine-tuning
    pad sequence to MAX_LEN length
    """
    return F.pad(seq, pad=(0, max_len - seq.shape[0]))
