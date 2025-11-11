import torch
import torch.nn as nn
import torch.nn.functional as F
from esm import pretrained
import math
import numpy as np
from Levenshtein import distance as levenshtein
import itertools
import os
import pandas as pd

class DualCrossAttentionUpdate(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.cross_attn1 = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.cross_attn2 = nn.MultiheadAttention(dim, num_heads, dropout=dropout)

        self.norm1_x1 = nn.LayerNorm(dim)
        self.norm1_x2 = nn.LayerNorm(dim)
        self.norm2_x1 = nn.LayerNorm(dim)
        self.norm2_x2 = nn.LayerNorm(dim)
        self.norm3_x1 = nn.LayerNorm(dim)
        self.norm3_x2 = nn.LayerNorm(dim)

        self.ffn1 = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x1, x2):
        x1_norm = self.norm1_x1(x1)
        x2_norm = self.norm1_x2(x2)

        attn_output1, _ = self.cross_attn1(
            query=x1_norm.transpose(0, 1),
            key=x2_norm.transpose(0, 1),
            value=x2_norm.transpose(0, 1)
        )
        updated_x1 = x1 + attn_output1.transpose(0, 1)

        attn_output2, _ = self.cross_attn2(
            query=x2_norm.transpose(0, 1),
            key=x1_norm.transpose(0, 1),
            value=x1_norm.transpose(0, 1)
        )
        updated_x2 = x2 + attn_output2.transpose(0, 1)

        ffn_output1 = self.ffn1(self.norm2_x1(updated_x1))
        ffn_output2 = self.ffn2(self.norm2_x2(updated_x2))
        
        output_x1 = self.norm3_x1(updated_x1 + ffn_output1)
        output_x2 = self.norm3_x2(updated_x2 + ffn_output2)
        
        return output_x1, output_x2


class Repair_model(nn.Module): 
    def __init__(self,esm_pretrain_model_HL, esm_pretrain_model_A, esm_alphabet, esm_last_layer, dropout, hidden_dim=128):
        super(Repair_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.esm_last_layer=esm_last_layer
        
        self.esm_alphabet =esm_alphabet

        self.esm_HL = esm_pretrain_model_HL
        self.esm_A = esm_pretrain_model_A
        
        for param in self.esm_HL.parameters():
            param.requires_grad = False
        for layer in self.esm_HL.layers[-3:]:
            for param in layer.parameters():
                param.requires_grad = True

        for param in self.esm_A.parameters():
            param.requires_grad = False
        for layer in self.esm_A.layers[-3:]:
            for param in layer.parameters():
                param.requires_grad = True

        esm_output_dim = self.esm_HL.embed_tokens.embedding_dim
        self.ff1 = nn.Sequential(
            nn.Linear(esm_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.ff2 = nn.Sequential(
            nn.Linear(esm_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.dual_cross_attention = DualCrossAttentionUpdate(dim=hidden_dim, num_heads=4, dropout=dropout)
        
        self.add_module("FC_{}1".format('affinity'), nn.Linear(hidden_dim*2, hidden_dim, bias=True))
        self.add_module("FC_{}2".format('affinity'), nn.Linear(hidden_dim, 1, bias=True))
        
    def forward(self, IGHL_tokens, IGA_tokens, IGHL_mask, IGA_mask):
        # Encoder
        IGHL_out = self.esm_HL(IGHL_tokens,repr_layers=[self.esm_last_layer]) 
        IGHL_emb = IGHL_out["representations"][self.esm_last_layer]
        IGA_out = self.esm_A(IGA_tokens,repr_layers=[self.esm_last_layer])
        IGA_emb = IGA_out["representations"][self.esm_last_layer]
        
        batch_size, seq_len1, seq_dim1 = IGHL_emb.shape
        _, seq_len2, seq_dim2 = IGA_emb.shape
        IGHL_emb = self.ff1(IGHL_emb.reshape(-1, seq_dim1)).view(batch_size, seq_len1, self.hidden_dim)
        IGA_emb = self.ff2(IGA_emb.reshape(-1, seq_dim2)).view(batch_size, seq_len2, self.hidden_dim)
        
        IGHL_emb, IGA_emb = self.dual_cross_attention(IGHL_emb, IGA_emb)
        
        # mean pooling
        IGHL_emb = self.masked_mean_pooling(IGHL_emb,IGHL_mask)
        IGA_emb = self.masked_mean_pooling(IGA_emb,IGA_mask)
        
        feature_embedding = torch.cat((IGHL_emb, IGA_emb), dim=1)
        
        emb = F.elu(self._modules["FC_{}1".format('affinity')](feature_embedding))
        output = self._modules["FC_{}2".format('affinity')](emb)
        return output
    
    def masked_mean_pooling(self,embeddings, mask):
        mask = mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        masked_embeddings = embeddings * mask  # [batch_size, seq_len, embed_dim]
        sum_embeddings = masked_embeddings.sum(dim=1)  # [batch_size, embed_dim]
        
        sum_mask = mask.sum(dim=1)  # [batch_size, 1]
        sum_mask = sum_mask.clamp(min=1e-9)  
        pooled_embeddings = sum_embeddings / sum_mask  # [batch_size, embed_dim]
    
        return pooled_embeddings


def model(model_path, device):
    esm_pretrain_model_HL, esm_alphabet = pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
    esm_pretrain_model_A, _ = pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
    model = Repair_model(esm_pretrain_model_HL, esm_pretrain_model_A, esm_alphabet,33, 0.2).to(device)
    state_dict = torch.load(model_path, device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, esm_alphabet


def diversity(seqs):
    num_seqs = len(seqs)
    total_dist = []
    for i in range(num_seqs):
        for j in range(num_seqs):
            x = seqs[i]
            y = seqs[j]
            if x == y:
                continue
            total_dist.append(levenshtein(x, y))
    return total_dist

def seq_recovery(wt_seq, generated_seqs):
    recovery_list = []
    wt_len = len(wt_seq)
    for seq in generated_seqs:
        compare_len = min(len(seq), wt_len)
        correct = sum(wt_seq[i] == seq[i] for i in range(compare_len))
        recovery = correct / wt_len
        recovery_list.append(recovery)
    return recovery_list

def aar(wt_seq, generated_seqs):
    wt_set = set(wt_seq)
    aar_list = []
    for seq in generated_seqs:
        gen_set = set(seq)
        overlap = len(gen_set & wt_set)
        aar_value = overlap / len(wt_set) if wt_set else 0.0
        aar_list.append(aar_value)
    return aar_list

class AbDesignEvalRunner:
    def __init__(self, model_path,device):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.score_model, self.esm_alphabet = model(model_path, self.device)

    def tokenize(self, seqs):
        CLS_IDX=0
        PADDING_idx=1
        EOS_IDX=2
        batch_converter = self.esm_alphabet.get_batch_converter()
        IGHL_seqs = [(pair_idx, wt_seq[0]) for pair_idx, wt_seq in enumerate(seqs)]
        IGA_seqs = [(pair_idx,wt_seq[1]) for pair_idx, wt_seq in enumerate(seqs)]
        _,_,IGHL_batch_tokens=batch_converter(IGHL_seqs)
        IGHL_mask = (IGHL_batch_tokens != PADDING_idx) & (IGHL_batch_tokens != EOS_IDX) & (IGHL_batch_tokens != CLS_IDX)
        _,_,IGA_batch_tokens=batch_converter(IGA_seqs)
        IGA_mask = (IGA_batch_tokens != PADDING_idx) & (IGA_batch_tokens != EOS_IDX) & (IGA_batch_tokens != CLS_IDX)
        
        return IGHL_batch_tokens.to(self.device), IGHL_mask.to(self.device), IGA_batch_tokens.to(self.device), IGA_mask.to(self.device)

    def _run_score_model(self, seqs):
        '''
            [(Abseq1, Agseq1), (Abseq2, Agseq2), ...]
        '''
        IGHL_tokens, IGHL_mask, IGA_tokens, IGA_mask = self.tokenize(seqs)    
        with torch.no_grad():
            scores = []
            batch_size=8
            num_batches = math.ceil(len(seqs) / batch_size)
            for batch_num in range(num_batches):
                start = batch_num * batch_size
                end = min((batch_num + 1) * batch_size, len(seqs))
                results = self.score_model(IGHL_tokens[start:end], 
                                           IGA_tokens[start:end], 
                                           IGHL_mask[start:end], 
                                           IGA_mask[start:end])
                scores.append(results)
        return torch.concat(scores, dim=0).squeeze().detach().cpu().numpy().tolist()
    
    def evaluate_sequences(self,generated_seqs,wt_seq):
        generated_seqs = generated_seqs[:128]
        wt_score = self._run_score_model(wt_seq)
        generated_seqs_scores = self._run_score_model(generated_seqs)
        
        delta_affinity = [mut_score-wt_score for mut_score in generated_seqs_scores]
        
        wt_ab = wt_seq[0][0]
        generated_ab = [ab for ab,ag in generated_seqs]
        
        ab_recovery = seq_recovery(wt_ab,generated_ab)
        ab_diversity = diversity(generated_ab)
        ab_aar = aar(wt_ab,generated_ab)
        
        res_dict={
            "delta_affinity":delta_affinity,
            "recovery":ab_recovery,
            "diversity":ab_diversity,
            "aar":ab_aar
            
        }

        return res_dict