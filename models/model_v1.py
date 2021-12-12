from models.layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATConv
from tqdm import tqdm
from transformers import BertModel, BertConfig
from csr_mhqa.utils import count_parameters


class CSGat(nn.Module):
    """
    Packing Query Version
    tx graph with GAT
    """
    def __init__(self, config):
        super(CSGat, self).__init__()
        self.config = config
        self.max_query_length = self.config.max_query_length

        self.bi_attention = BiAttention(input_dim=config.input_dim,
                                        memory_dim=config.input_dim,
                                        hid_dim=config.hidden_dim,
                                        dropout=config.bi_attn_drop)
        self.bi_attn_linear = nn.Linear(config.hidden_dim * 4, config.hidden_dim)

        self.hidden_dim = config.hidden_dim
        self.to_hidden_dim = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.sent_lstm = LSTMWrapper(input_dim=config.hidden_dim,
                                     hidden_dim=config.hidden_dim,
                                     n_layer=1,
                                     dropout=config.lstm_drop)

        self.random_emb = torch.empty(6, self.hidden_dim)
        nn.init.xavier_uniform_(self.random_emb, gain=nn.init.calculate_gain('relu'))

        # use DGL gat impl
        self.gat_layers = nn.ModuleList()
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            config.hidden_dim, config.hidden_dim, config.gnn_attn_head[0],
            feat_drop=config.gnn_feat_drop, residual=False))
        # hidden layers
        for l in range(1, config.gnn_layer - 2):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                config.hidden_dim * config.gnn_attn_head[l - 1], config.hidden_dim, config.gnn_attn_head[l],
                feat_drop=config.gnn_feat_drop, residual=config.gnn_residual))
        # output projection
        self.gat_layers.append(GATConv(
            config.hidden_dim * config.gnn_attn_head[-2], config.hidden_dim, config.gnn_attn_head[-1],
            feat_drop=config.gnn_feat_drop, residual=config.gnn_residual))

        self.ctx_attention = GatedAttention(input_dim=config.hidden_dim*2,
                                            memory_dim=config.hidden_dim if config.q_update else config.hidden_dim*2,
                                            hid_dim=self.config.ctx_attn_hidden_dim,
                                            dropout=config.bi_attn_drop,
                                            gate_method=self.config.ctx_attn)

        q_dim = self.hidden_dim if config.q_update else config.input_dim

        self.predict_layer = PredictionLayer(self.config, q_dim)

    def forward(self, batch, return_yp):
        query_mapping = batch['query_mapping']
        context_encoding = batch['context_encoding']

        # extract query encoding
        trunc_query_mapping = query_mapping[:, :self.max_query_length].contiguous()
        trunc_query_state = (context_encoding * query_mapping.unsqueeze(2))[:, :self.max_query_length, :].contiguous()
        # bert encoding query vec
        query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        attn_output, trunc_query_state = self.bi_attention(context_encoding,
                                                           trunc_query_state,
                                                           trunc_query_mapping)

        input_state = self.bi_attn_linear(attn_output)  # N x L x d
        input_state = self.sent_lstm(input_state, batch['context_lens'])  # todo: what does sent_lstm do to vectors?

        # if self.config.q_update:
        #     query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)
        query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        para_logits, sent_logits = [], []
        para_predictions, sent_predictions, ent_predictions = [], [], []

        # observe the above vectors, todo: concatenate vectors into graph features

        sent_start_mapping = batch['sent_start_mapping']
        sent_end_mapping = batch['sent_end_mapping']
        para_start_mapping = batch['para_start_mapping']
        para_end_mapping = batch['para_end_mapping']
        ent_start_mapping = batch['ent_start_mapping']
        ent_end_mapping = batch['ent_end_mapping']

        def get_span_pooled_vec(state_input, mapping):
            mapping_state = state_input.unsqueeze(2) * mapping.unsqueeze(3)
            mapping_sum = mapping.sum(dim=1)
            mapping_sum = torch.where(mapping_sum == 0, torch.ones_like(mapping_sum), mapping_sum)
            mean_pooled = mapping_state.sum(dim=1) / mapping_sum.unsqueeze(-1)

            return mean_pooled
        # cat start and end embedding as representation
        para_start_output = torch.bmm(para_start_mapping, input_state[:, :, self.hidden_dim:])   # N x max_para x d
        para_end_output = torch.bmm(para_end_mapping, input_state[:, :, :self.hidden_dim])       # N x max_para x d
        para_state = self.to_hidden_dim(torch.cat([para_start_output, para_end_output], dim=-1))  # 2d -> N x max_para x d

        sent_start_output = torch.bmm(sent_start_mapping, input_state[:, :, self.hidden_dim:])   # N x max_sent x d
        sent_end_output = torch.bmm(sent_end_mapping, input_state[:, :, :self.hidden_dim])       # N x max_sent x d
        sent_state = self.to_hidden_dim(torch.cat([sent_start_output, sent_end_output], dim=-1))  # 2d -> N x max_sent x d

        ent_start_output = torch.bmm(ent_start_mapping, input_state[:, :, self.hidden_dim:])   # N x max_ent x d
        ent_end_output = torch.bmm(ent_end_mapping, input_state[:, :, :self.hidden_dim])       # N x max_ent x d
        ent_state = self.to_hidden_dim(torch.cat([ent_start_output, ent_end_output], dim=-1))  # 2d -> N x max_ent x d

        # to get whole graph feature, stack states of query, paras, sents, ents and edge nodes in a whole
        # didn't think of a good way to do this in parallel
        graphs = batch['graph']
        all_features = []
        for g_idx, g in enumerate(graphs):
            g_ed_node_feats = []
            tmp_ed_node_feat = []
            pse_num = []  # para, sent, entity num
            tmp_pse_num = 0  # tmp para, sent, entity num
            for i, span in enumerate(g.ndata['pos'][1:]):
                if span[0] < 0 and span[0] == span[1]:
                    tmp_ed_node_feat.append(self.random_emb[span[0]+6])
                    if g.ndata['pos'][i][0] > 0:
                        pse_num.append(tmp_pse_num)
                        tmp_pse_num = 0
                else:
                    tmp_pse_num += 1
                    if g.ndata['pos'][i][0] < 0:
                        g_ed_node_feats.append(torch.stack(tmp_ed_node_feat))
                        tmp_ed_node_feat = []
            # g_feats = query_vec + para_vec + g_ed_node_feats[0] + sent_vec + g_ed_node_feats[1] + ent_vec
            g_feats = torch.cat((query_vec[g_idx:g_idx+1], para_state[g_idx][:pse_num[0]], g_ed_node_feats[0],
                                 sent_state[g_idx][:pse_num[1]], g_ed_node_feats[1],
                                 ent_state[g_idx][:pse_num[2]], g_ed_node_feats[2]), dim=0)
            all_features.append(g_feats)
        all_features = torch.stack(all_features)

        # Graph module
        graph_list = []
        for i, g in enumerate(graphs):
            g = g.remove_self_loop()
            graph_list.append(g.add_self_loop())
        bg = dgl.batch(graph_list)

        for l in range(self.gnn_layer):
            all_features = self.gat_layers[l](bg, all_features).flatten(1)
        # output projection
        all_features = self.gat_layers[-1](bg, all_features).flatten(1)

        # process graph features










        for l in range(self.config.num_gnn_layers):
            new_input_state, graph_state, graph_mask, sent_state, query_vec, para_logit, para_prediction, \
            sent_logit, sent_prediction, ent_logit = self.graph_blocks[l](batch, input_state, query_vec)

            para_logits.append(para_logit)
            sent_logits.append(sent_logit)
            para_predictions.append(para_prediction)
            sent_predictions.append(sent_prediction)
            ent_predictions.append(ent_logit)

        input_state, _ = self.ctx_attention(input_state, graph_state, graph_mask.squeeze(-1))
        predictions = self.predict_layer(batch, input_state, sent_logits[-1], packing_mask=query_mapping, return_yp=return_yp)

        if return_yp:
            start, end, q_type, yp1, yp2 = predictions
            return start, end, q_type, para_predictions[-1], sent_predictions[-1], ent_predictions[-1], yp1, yp2
        else:
            start, end, q_type = predictions
            return start, end, q_type, para_predictions[-1], sent_predictions[-1], ent_predictions[-1]
