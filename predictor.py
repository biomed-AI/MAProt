import torch
import torch.nn as nn
import torch.nn.functional as F
from common_utils.protein_mpnn_utils import ProteinMPNN, _scores
import esm
from esm import ESM2, Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
from common_utils.esm_loader import load_esm_saprot

from transformers import EsmTokenizer, EsmForMaskedLM

class DPOLoss(nn.Module):
    def __init__(self, beta):
        super(DPOLoss, self).__init__()
        self.beta = beta
        self.sigmoid = nn.Sigmoid()

    def forward(self, S_prefer,S_disprefer,log_probs_policy,log_probs_ref,mask,noise=None):
        dispreferred_scores = _scores(S_disprefer, log_probs_policy, mask) 
        preferred_scores = _scores(S_prefer, log_probs_policy, mask) 

        ref_dispreferred_scores = _scores(S_disprefer, log_probs_ref, mask) 
        ref_preferred_scores = _scores(S_prefer, log_probs_ref, mask) 
        
        score_diff = -(preferred_scores - dispreferred_scores)
        ref_score_diff = -(ref_preferred_scores - ref_dispreferred_scores)
        preference_prob = self.sigmoid(torch.clamp(self.beta *(score_diff-ref_score_diff), -10, 10))

        per_sample_loss = -torch.log(preference_prob + 1e-8)
        if noise is not None:
            assert noise.shape[0] == per_sample_loss.shape[0],f'wrong noise size noise:{noise.shape}  S:{S_prefer.shape}'
            per_sample_loss = per_sample_loss * noise 
        dpo_loss = per_sample_loss.mean()

        return dpo_loss

class MultiAgentDPOLoss(nn.Module):
    def __init__(self, beta):
        super(MultiAgentDPOLoss, self).__init__()
        self.dpo_loss_func = DPOLoss(beta)

    def forward(self, each_agent_encoding_pair_list,log_probs_policy_tuple,log_probs_ref_list_tuple,noise=None):
        total_loss = 0
        for encoding_pair,log_probs_policy,log_probs_ref in zip(
                each_agent_encoding_pair_list,
                log_probs_policy_tuple,
                log_probs_ref_list_tuple
            ):
            S_prefer,S_disprefer,mask = encoding_pair
            total_loss+=self.dpo_loss_func(S_prefer,S_disprefer,log_probs_policy,log_probs_ref,mask,noise)
        
        return total_loss




# multi agent package
class MultiAgentPredictor(nn.Module):
    def __init__(self, cfg,esm_pretrain_model, esm_alphabet,saprot_pretrain_model,saprot_tokenizer):
        super().__init__()
        self.mpnn = ProteinMPNN(ca_only=cfg.ca_only, num_letters=21, 
                node_features=cfg.hidden_dim, edge_features=cfg.hidden_dim, hidden_dim=cfg.hidden_dim, 
                num_encoder_layers=cfg.num_layers, num_decoder_layers=cfg.num_layers, augment_eps=cfg.backbone_noise, 
                k_neighbors=cfg.num_edges)
        
        self.esm = esm_pretrain_model
        self.esm_alphabet = esm_alphabet
        
        self.saprot = saprot_pretrain_model
        self.saprot_tokenizer = saprot_tokenizer
        
        # Freeze all parameters of the ESM model and unfreeze the last three layers
        for param in self.esm.parameters():
            param.requires_grad = False
        for layer in self.esm.layers[-3:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Freeze all parameters of the SaProt model and unfreeze the last three layers
        for param in self.saprot.parameters():
            param.requires_grad = False
        for layer in self.saprot.esm.encoder.layer[-3:]:
            for param in layer.parameters():
                param.requires_grad = True
        
    def forward(self, esm_S_wt, saprot_S_wt, mpnn_S_wt, mpnn_mask, X, chain_M, residue_idx, chain_encoding_all):
        log_probs_mpnn,logits_mpnn = self.mpnn.deterministic_forward(X, mpnn_S_wt, mpnn_mask, chain_M, residue_idx, chain_encoding_all)
        
        logits_esm = self.esm(esm_S_wt)["logits"]
        log_probs_esm = torch.log_softmax(logits_esm, dim=-1)
        
        logits_saprot = self.saprot(saprot_S_wt)["logits"]
        log_probs_saprot = torch.log_softmax(logits_saprot, dim=-1)
        
        return (log_probs_mpnn,log_probs_esm,log_probs_saprot),(logits_mpnn,logits_esm,logits_saprot)
        

        
class MultiAgentModelManager(object):

    def __init__(self, config,model_factory):
        super().__init__()
        esm_pretrain_model, esm_alphabet = pretrained.load_model_and_alphabet(config.ESM_pretrain_model)
        
        saprot_tokenizer = EsmTokenizer.from_pretrained(config.SaProt_pretrain_model)
        saprot_pretrain_model = EsmForMaskedLM.from_pretrained(config.SaProt_pretrain_model)
        
        self.model = model_factory(config,esm_pretrain_model,esm_alphabet,saprot_pretrain_model, saprot_tokenizer)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.98))


    def get(self,):
        return self.model, self.optimizer

    def to(self, device):
        self.model.to(device)
        return self

    def load_mpnn_state_dict(self, state_dict):
        self.model.mpnn.load_state_dict(state_dict['model_state_dict'], strict=False)
        
    def state_dict(self):
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'], strict=False)
        self.optimizer.load_state_dict(state_dict['optimizer'])
        
    def load_state_dict_inference(self, state_dict):
        self.model.load_state_dict(state_dict['model'], strict=False)
        

