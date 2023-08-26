from typing import Optional, Tuple
from fastNLP.modules import Seq2SeqEncoder, Seq2SeqDecoder, State
import torch
import torch.nn.functional as F
from torch import nn
from src.model.modeling_bart import (PretrainedBartModel, BartEncoder,
                                     BartDecoder, BartModel,
                                     BartClassificationHead,
                                     _make_linear_from_emb,
                                     _prepare_bart_decoder_inputs)
from transformers import BartTokenizer

from src.model.config import MultiModalBartConfig
from src.model.modules import MultiModalBartEncoder, MultiModalBartDecoder_span, Span_loss
import numpy as np
import torch.nn as nn
import math
from src.model.GCN import GCN
# from src.otk.layers import OTKernel
# from src.otk.models import SeqAttention
import random
import math
import copy
from src.model.GAT import GAT
import numpy as np
from src.model.layers import GraphLearner
import os
import torch.cuda as cuda

from src.otk.layers import OTKernel


# |-------------------------------------------------------------------------------|
# |                  !!!!!!!!!!!!!!!!!!!!!!!                                      |
# |-------------------------------------------------------------------------------|


class MultiModalBartModel_AESC(PretrainedBartModel):
    def build_model(self,
                    args,
                    bart_model,
                    tokenizer,
                    label_ids,
                    config,
                    decoder_type=None,
                    copy_gate=False,
                    use_encoder_mlp=False,
                    use_recur_pos=False,
                    tag_first=False):
        if args.bart_init:
            model = BartModel.from_pretrained(bart_model)
            num_tokens, _ = model.encoder.embed_tokens.weight.shape
            print('num_tokens', num_tokens)

            model.resize_token_embeddings(
                len(tokenizer.unique_no_split_tokens) + num_tokens)
            encoder = model.encoder
            decoder = model.decoder

            padding_idx = config.pad_token_id
            encoder.embed_tokens.padding_idx = padding_idx

            # if use_recur_pos:
            #     decoder.set_position_embedding(label_ids[0], tag_first)

            _tokenizer = BartTokenizer.from_pretrained(bart_model)

            for token in tokenizer.unique_no_split_tokens:
                if token[:2] == '<<':  # 特殊字符
                    index = tokenizer.convert_tokens_to_ids(
                        tokenizer._base_tokenizer.tokenize(token))
                    if len(index) > 1:
                        raise RuntimeError(f"{token} wrong split")
                    else:
                        index = index[0]
                    assert index >= num_tokens, (index, num_tokens, token)
                    indexes = _tokenizer.convert_tokens_to_ids(
                        _tokenizer.tokenize(token[2:-2]))
                    embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                    for i in indexes[1:]:
                        embed += model.decoder.embed_tokens.weight.data[i]
                    embed /= len(indexes)
                    model.decoder.embed_tokens.weight.data[index] = embed
        else:
            raise RuntimeError("error init!!!!!!!")

        multimodal_encoder = MultiModalBartEncoder(config, encoder,
                                                   tokenizer.img_feat_id,
                                                   tokenizer.cls_token_id)
        return multimodal_encoder, decoder

    def __init__(self, config: MultiModalBartConfig, args, bart_model,
                 tokenizer, label_ids):
        super().__init__(config)
        self.config = config
        self.mydevice = args.device
        label_ids = sorted(label_ids)
        multimodal_encoder, share_decoder = self.build_model(
            args, bart_model, tokenizer, label_ids, config)
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        self.causal_mask = causal_mask.triu(diagonal=1)
        self.encoder = multimodal_encoder
        only_sc = False
        need_tag = True  # if predict the sentiment or not
        self.gcn_on = args.gcn_on
        self.sentinet_on = args.sentinet_on
        self.dep_mode = args.dep_mode
        self.nn_attention_on = args.nn_attention_on
        self.nn_attention_mode = args.nn_attention_mode
        self.gcn_dropout = args.gcn_dropout
        self.gcn_proportion = args.gcn_proportion

        self.decoder = MultiModalBartDecoder_span(self.config,
                                                  tokenizer,
                                                  share_decoder,
                                                  tokenizer.pad_token_id,
                                                  label_ids,
                                                  self.causal_mask,
                                                  self.gcn_on,
                                                  need_tag=need_tag,
                                                  only_sc=False)
        self.span_loss_fct = Span_loss()

        # add
        self.noun_linear = nn.Linear(768, 768)
        self.multi_linear = nn.Linear(768, 768)
        self.att_linear = nn.Linear(768 * 2, 1)
        self.attention = Attention(4, 768, 768)
        self.linear = nn.Linear(768 * 2, 1)
        self.linear2 = nn.Linear(768 * 2, 1)

        self.alpha_linear1 = nn.Linear(768, 768)
        self.alpha_linear2 = nn.Linear(768, 768)

        self.senti_linear = nn.Linear(768, 768)
        self.context_linear = nn.Linear(768, 768)
        self.mix_linear = nn.Linear(768 * 2, 768)

        # GCN and GAT
        self.senti_gcn = GCN(768, 768, 768, dropout=self.gcn_dropout)
        self.context_gcn = GCN(768, 768, 768, dropout=self.gcn_dropout)
        self.gat = GAT(768, 768, 0.2, 0.2, n_heads=1)

        # 可学习参数
        self.graph_skip_conn = nn.Parameter(torch.tensor(0.5))

        self.gat_linear = nn.Linear(768, 768)
        self.pos_embedding = POS_embedding(32, 1)

        self.senti_value_linear = nn.Linear(1, 768)
        self.dep_linear1 = nn.Linear(768, 768)
        self.dep_linear2 = nn.Linear(768, 768)
        self.dep_att_linear = nn.Linear(768 * 2, 1)
        # 创建一个GraphLearner实例
        self.graph_learner = GraphLearner(input_size=768, hidden_size=128, graph_type='KNN', top_k=10, num_pers=4,
                                          metric_type="attention")
        self.self_attention = MultiHeadAttention(768, 768, 768, 8)
        self.cross_attention = MultiHeadAttention(768, 768, 768, 8)

    def get_noun_embed(self, feature, noun_mask):
        # print(feature.shape,noun_mask.shape)
        noun_mask = noun_mask.cpu()
        noun_num = [x.numpy().tolist().count(1) for x in noun_mask]
        noun_position = [np.where(np.array(x) == 1)[0].tolist() for x in noun_mask]
        for i, x in enumerate(noun_position):
            assert len(x) == noun_num[i]
        max_noun_num = max(noun_num)

        # pad
        for i, x in enumerate(noun_position):
            if len(x) < max_noun_num:
                noun_position[i] += [0] * (max_noun_num - len(x))
        #       TODO
        # self.mydevice = torch.device("cuda:5")
        noun_position = torch.tensor(noun_position).to(self.mydevice)
        noun_embed = torch.zeros(feature.shape[0], max_noun_num, feature.shape[-1]).to(self.mydevice)
        for i in range(len(feature)):
            noun_embed[i] = torch.index_select(feature[i], dim=0, index=noun_position[i])
            noun_embed[i, noun_num[i]:] = torch.zeros(max_noun_num - noun_num[i], feature.shape[-1])
        return noun_embed

    def prepare_state(self,
                      input_ids,
                      image_features,
                      # noun_ids,
                      noun_mask,
                      attention_mask=None,
                      dependency_matrix=None,
                      sentiment_value=None,
                      pos_ids=None,
                      raw_token_ids=None,
                      first=None):
        dict = self.encoder(input_ids=input_ids,
                            image_features=image_features,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            return_dict=True)
        encoder_outputs = dict.last_hidden_state
        # print("-----------------encoder_outputs.shape-----------------")
        # print(encoder_outputs.shape)
        # otklayer = nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.ReLU(),
        #     OTKernel(in_dim=768, out_size=encoder_outputs.shape[1],heads=1)
        # )
        # otklayer = otklayer.to('cuda:0')  # Move otklayer to GPU 5
        # otk_outputs = otklayer(encoder_outputs)

        # otk_model = SeqAttention(
        #     encoder_outputs.shape[1], nclass, [hidden_size], [1], [1], [0.6], out_size=10, heads=4
        # )
        # output = otk_model(input)

        hidden_states = dict.hidden_states
        encoder_mask = attention_mask
        src_embed_outputs = hidden_states[0]

        if self.nn_attention_on:
            # 获取名词的embedding
            noun_embed = self.get_noun_embed(encoder_outputs, noun_mask)
            encoder_outputs = self.noun_attention(encoder_outputs, noun_embed, mode=self.nn_attention_mode)

        # print(encoder_outputs.shape)
        # encoder_outputs = otklayer(encoder_outputs)
        # gcn
        senti_feature, context_feature, mix_feature = None, None, None
        if self.sentinet_on and self.gcn_on:
            # 2023.7.17 测试不适用senticNet融合
            mix_feature = self.multimodal_GCN(encoder_outputs, dependency_matrix, attention_mask, noun_mask,
                                              sentiment_value)
            # mix_feature = self.multimodal_GCN(encoder_outputs, dependency_matrix, attention_mask, noun_mask)
        elif self.gcn_on:
            mix_feature = self.multimodal_GCN(encoder_outputs, dependency_matrix, attention_mask, noun_mask)

        # |-------------------------------------------------------------------------------|
        # |                                 attention                                     |
        # |-------------------------------------------------------------------------------|
        # print("----------------encoder_outputs----------------------")
        # print(encoder_outputs.shape) 32 * 99 * 768
        # print("----------------mix_feature----------------------")
        # print(mix_feature.shape) 32 * 99 * 768
        # self_encoder_outputs = self.self_attention(encoder_outputs, encoder_outputs)
        # mix_feature = self.self_attention(self_encoder_outputs, mix_feature)
        # mix_feature = self.self_attention(otk_outputs, mix_feature)

        state = BartState(
            encoder_outputs,
            encoder_mask,
            input_ids[:, 51:],  # the text features start from index 38, the front are image features.
            first,
            src_embed_outputs,
            mix_feature
        )
        # setattr(state, 'tgt_seq_len', tgt_seq_len)
        return state

    def noun_attention(self, encoder_outputs, noun_embed, mode='multi-head'):
        if mode == 'cat':
            multi_features_rep = encoder_outputs.unsqueeze(2).repeat(1, 1, noun_embed.shape[1], 1)
            noun_features_rep = noun_embed.unsqueeze(1).repeat(1, encoder_outputs.shape[1], 1, 1)
            noun_features_rep = self.noun_linear(noun_features_rep)
            multi_features_rep = self.multi_linear(multi_features_rep)
            concat_features = torch.tanh(torch.cat([noun_features_rep, multi_features_rep], dim=-1))
            att = torch.softmax(self.att_linear(concat_features).squeeze(-1), dim=-1)
            att_features = torch.matmul(att, noun_embed)

            alpha = torch.sigmoid(
                self.linear(torch.cat([self.alpha_linear1(encoder_outputs), self.alpha_linear2(att_features)], dim=-1)))
            alpha = alpha.repeat(1, 1, 768)

            encoder_outputs = torch.mul(1 - alpha, encoder_outputs) + torch.mul(alpha, att_features)

            return encoder_outputs
        elif mode == 'none':
            return encoder_outputs
        elif mode == 'multi-head':
            # 多头注意力
            att_features = self.attention(encoder_outputs, noun_embed, noun_embed)
            alpha = torch.sigmoid(self.linear(torch.cat([encoder_outputs, att_features], dim=-1)))
            alpha = alpha.repeat(1, 1, 768)
            encoder_outputs = torch.mul(1 - alpha, encoder_outputs) + torch.mul(alpha, att_features)
            return encoder_outputs
        elif mode == 'cos_':
            multi_features_rep = encoder_outputs.unsqueeze(1).repeat(1, noun_embed.shape[1], 1, 1)
            noun_features_rep = noun_embed.unsqueeze(2).repeat(1, 1, encoder_outputs.shape[1], 1)
            att = torch.cosine_similarity(multi_features_rep, noun_features_rep, dim=-1)
            att = att.max(1)[1]
            att_features = torch.zeros(encoder_outputs.shape).to(self.mydevice)
            for i in range(noun_embed.shape[0]):
                att_features[i] = torch.index_select(noun_embed[i], 0, att[i])

            alpha = torch.sigmoid(self.linear(torch.cat([encoder_outputs, att_features], dim=-1)))
            alpha = alpha.repeat(1, 1, 768)
            encoder_outputs = torch.mul(alpha, encoder_outputs) + torch.mul(1 - alpha, att_features)
            return encoder_outputs

    def multimodal_GCN(self, encoder_outputs, dependency_matrix, attention_mask, noun_mask, sentiment_value=None,
                       threshold=0.8, dropout=0.8):
        # print(dependency_matrix.shape) 16 * 43 * 43
        # print(dependency_matrix[0][1]) 测试过了，和前面的一摸一样
        new_dependency_matrix = torch.zeros(
            [encoder_outputs.shape[0], encoder_outputs.shape[1], encoder_outputs.shape[1]], dtype=torch.float).to(
            encoder_outputs.device)
        img_feature = encoder_outputs[:, :51, :]
        text_feature = encoder_outputs[:, 51:, :]

        # dep_list = ['text_cosine', 'text_cat_sim', 'text_cos_img_noun_sim']
        if self.dep_mode == 'text_cosine':
            # 以token之间的相似度作为依赖值
            text_feature_extend1 = text_feature.unsqueeze(1).repeat(1, text_feature.shape[1], 1, 1)
            text_feature_extend2 = text_feature.unsqueeze(2).repeat(1, 1, text_feature.shape[1], 1)
            text_sim = torch.cosine_similarity(text_feature_extend1, text_feature_extend2, dim=-1)
            new_dependency_matrix[:, 51:, 51:] = dependency_matrix * text_sim
        elif self.dep_mode == 'text_cat_sim':
            text_feature_extend1 = text_feature.unsqueeze(1).repeat(1, text_feature.shape[1], 1, 1)
            text_feature_extend2 = text_feature.unsqueeze(2).repeat(1, 1, text_feature.shape[1], 1)
            text_feature_extend1 = self.dep_linear1(text_feature_extend1)
            text_feature_extend2 = self.dep_linear2(text_feature_extend2)
            att = torch.softmax(self.dep_att_linear(torch.tanh(torch.cat(
                [text_feature_extend1, text_feature_extend2], dim=-1
            ))).squeeze(-1), dim=-1)
            new_dependency_matrix[:, 51:, 51:] = dependency_matrix * att
        elif self.dep_mode == 'text_cos_img_noun_sim':
            # 计算图像patch和文本token的关联度作为依赖矩阵中图片的值
            img_feature_extend = img_feature.unsqueeze(2).repeat(1, 1, text_feature.shape[1], 1)
            text_feature_extend = text_feature.unsqueeze(1).repeat(1, img_feature.shape[1], 1, 1)
            sim = torch.cosine_similarity(img_feature_extend, text_feature_extend, dim=-1)

            # 图像只与名词挂钩
            noun_mask = noun_mask[:, 51:].unsqueeze(1).repeat(1, sim.shape[1], 1)
            sim = sim * noun_mask
            new_dependency_matrix[:, :51, 51:] = sim
            new_dependency_matrix[:, 51:, :51] = torch.transpose(sim, 1, 2)

            # 以token之间的相似度作为依赖值
            text_feature_extend1 = text_feature.unsqueeze(1).repeat(1, text_feature.shape[1], 1, 1)
            text_feature_extend2 = text_feature.unsqueeze(2).repeat(1, 1, text_feature.shape[1], 1)
            text_sim = torch.cosine_similarity(text_feature_extend1, text_feature_extend2, dim=-1)
            new_dependency_matrix[:, 51:, 51:] = dependency_matrix * text_sim

        # new_dependency_matrix[:,51:,51:]=dependency_matrix
        # 填充的文本区域的，从51行51列开始，即从右下角的区域开始填充
        for i in range(new_dependency_matrix.shape[1]):
            new_dependency_matrix[:, i, i] = 1

        # GCN部分
        context_dependency_matrix = new_dependency_matrix.clone().detach()
        # print(context_dependency_matrix.shape) 16 * 94 *94

        # |-------------------------------------------------------------------------------|
        # |                                                                               |
        # |-------------------------------------------------------------------------------|

        # 2023.7.28 IB理论图
        # 使用GraphLearner处理Tensor
        new_node_features, learned_adj = self.graph_learner(encoder_outputs)

        # 填充图像区域情感值
        if self.sentinet_on:
            # print("-------before sentiment_value-------") 16 * 43
            sentiment_value = nn.ZeroPad2d(padding=(51, 0, 0, 0))(sentiment_value)
            sentiment_value = sentiment_value.unsqueeze(-1)
            # print("-------after sentiment_value-------") 16 * 94 * 1
            # 其中的线性层即代表可学习参数
            sentiment_feature = self.senti_value_linear(sentiment_value)
            context_feature = self.context_linear(encoder_outputs + sentiment_feature)
            # |-------------------------------------------------------------------------------|
            # |                                                                               |
            # |-------------------------------------------------------------------------------|

            context_feature = self.context_gcn(new_node_features, learned_adj, attention_mask)
            context_feature = self.senti_gcn(context_feature, context_dependency_matrix, attention_mask)

        else:
            # print(encoder_outputs.shape) 16 * seq * 768
            # context_feature = self.context_gcn(encoder_outputs, context_dependency_matrix, attention_mask)

            # 两层GCN
            context_feature = self.gat(new_node_features, learned_adj)
            # context_dependency_matrix = self.graph_skip_conn * context_dependency_matrix + (1 - self.graph_skip_conn) * learned_adj
            context_feature = self.senti_gcn(context_feature, context_dependency_matrix, attention_mask)

        mix_feature = self.gcn_proportion * context_feature + encoder_outputs

        return mix_feature

    def forward(
            self,
            input_ids,
            image_features,
            sentiment_value,
            noun_mask,
            attention_mask=None,
            dependency_matrix=None,
            aesc_infos=None,
            encoder_outputs: Optional[Tuple] = None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        state = self.prepare_state(input_ids, image_features, noun_mask, attention_mask, dependency_matrix,
                                   sentiment_value)
        spans, span_mask = [
            aesc_infos['labels'].to(input_ids.device),
            aesc_infos['masks'].to(input_ids.device)
        ]
        logits = self.decoder(spans, state, sentiment_value)
        # logits = self.decoder(spans, state)

        loss = self.span_loss_fct(spans[:, 1:], logits, span_mask[:, 1:])
        return loss


class BartState(State):
    def __init__(self, encoder_output, encoder_mask, src_tokens, first,
                 src_embed_outputs, mix_feature):
        super().__init__(encoder_output, encoder_mask)
        self.past_key_values = None
        self.src_tokens = src_tokens
        self.first = first
        self.src_embed_outputs = src_embed_outputs
        self.mix_feature = mix_feature

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.src_tokens = self._reorder_state(self.src_tokens, indices)
        if self.first is not None:
            self.first = self._reorder_state(self.first, indices)
        self.src_embed_outputs = self._reorder_state(self.src_embed_outputs,
                                                     indices)

        if self.mix_feature is not None:
            self.mix_feature = self._reorder_state(self.mix_feature,
                                                   indices)
        if self.past_key_values is not None:
            new = []
            for layer in self.past_key_values:
                new_layer = {}
                for key1 in list(layer.keys()):
                    new_layer_ = {}
                    for key2 in list(layer[key1].keys()):
                        if layer[key1][key2] is not None:
                            layer[key1][key2] = self._reorder_state(
                                layer[key1][key2], indices)
                            # print(key1, key2, layer[key1][key2].shape)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new


class Attention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(Attention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[: -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v):
        key = self.key_layer(k)
        query = self.query_layer(q)
        value = self.value_layer(v)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim=-1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[: -2] + (self.all_head_size,)
        context = context.view(*new_size)
        return context


'''
词性特征表示
'''

from pytorch_transformers.modeling_bert import BertSelfAttention, BertConfig
import torch.nn as nn
import torch

'''
词性特征表示
测试，使用spacy3.6
'''

# Predefined list of POS tags (you can extend or modify this list based on your needs)
POS_TAGS = [
    "<pad>", "NN", "VB", "DT", "IN", "JJ", "NNS", "VBD", "VBG", "VBN", "VBP", "VBZ",
    "RB", "CC", "TO", "PRP", "MD", "POS", "PRP$", "WDT", "JJR", "WP", "WRB", "JJS",
    "CD", "EX", "RP", "WP$", "NNP", "RBR", "RBS", "PDT", "FW", "UH", "SYM", "LS", "NNPS"
]


class POS_embedding(nn.Module):
    def __init__(self, pos_embedding_size=32, pos_att_head=1):
        super(POS_embedding, self).__init__()
        self.tag_map = {tag: i for i, tag in enumerate(POS_TAGS)}  # Convert to tag -> id
        self.embeddings = nn.Embedding(len(self.tag_map), pos_embedding_size)
        pos_attention_config = BertConfig(hidden_size=pos_embedding_size, num_attention_heads=pos_att_head)
        self.pos_attention = BertSelfAttention(pos_attention_config)

    def forward(self, pos_ids):
        attention_mask = torch.ones_like(pos_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        pos_output = self.embeddings(pos_ids)
        pos_output = self.pos_attention(pos_output, extended_attention_mask)[0]
        return pos_output

# class MultiHeadSelfAttention(nn.Module):
#     def __init__(self, embed_size, num_heads, dropout=0.5):
#         super(MultiHeadSelfAttention, self).__init__()
#         self.embed_size = embed_size
#         self.num_heads = num_heads
#         self.head_dim = embed_size // num_heads
#
#         assert (
#             self.head_dim * num_heads == embed_size
#         ), "Embedding size needs to be divisible by num_heads"
#
#         self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
#         self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
#         self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
#         self.fc_out = nn.Linear(self.embed_size, self.embed_size)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, values, keys, queries, mask=None):
#         N = queries.shape[0]
#         value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
#
#         # Split embedding into self.num_heads different pieces
#         values = values.reshape(N, value_len, self.num_heads, self.head_dim)
#         keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
#         queries = queries.reshape(N, query_len, self.num_heads, self.head_dim)
#
#         values = self.values(values)
#         keys = self.keys(keys)
#         queries = self.queries(queries)
#
#         # Scaled dot-product attention
#         inner_product = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
#         keys_dim = keys.shape[-1]
#         scaled_inner_product = inner_product / (keys_dim ** 0.5)
#
#         if mask is not None:
#             scaled_inner_product = scaled_inner_product.masked_fill(mask == 0, float("-inf"))
#
#         attention = F.softmax(scaled_inner_product, dim=-1)
#         attention = self.dropout(attention)
#
#         out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
#             N, query_len, self.embed_size
#         )
#         out = self.fc_out(out)
#         return out
class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
        mask --- [N, T_k]
    output:
        out --- [N, T_q, num_units]
        scores -- [h, N, T_q, T_k]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key, mask=None):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        ## score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)

        ## mask
        if mask is not None:
            ## mask:  [N, T_k] --> [h, N, T_q, T_k]
            mask = mask.unsqueeze(1).unsqueeze(0).repeat(self.num_heads, 1, querys.shape[2], 1)
            scores = scores.masked_fill(mask, -np.inf)
        scores = F.softmax(scores, dim=3)

        ## out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out