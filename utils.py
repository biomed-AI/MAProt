import os
import shutil
import random
import numpy as np

import torch
from sklearn.metrics import roc_auc_score
from common_utils.protein_mpnn_utils import _scores
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import roc_auc_score

from datetime import datetime
import time
import re
import logging
from tqdm import tqdm

import pandas as pd
from biotite.sequence.io import fasta
import pyrosetta
pyrosetta.init(extra_options="-out:level 100")
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
from pyrosetta import *


ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
def featurize_mpnn(batch, device):
    B = batch['structure'].shape[0]
    L_max = max([len(x) for x in batch['aa_seq_wt']])
    X = batch['structure'][:, :L_max, :, :].to(dtype=torch.float32, device=device)
    S_prefer = np.zeros([B, L_max], dtype=np.int32) #sequence AAs integers
    S_disprefer = np.zeros([B, L_max], dtype=np.int32) #sequence AAs integers
    S_wt = np.zeros([B, L_max], dtype=np.int32)
    mask = np.zeros([B, L_max], dtype=np.int32)
    residue_idx = -100*np.ones([B, L_max], dtype=np.int32)
    for i, seq in enumerate(batch['aa_seq_wt']):
        S_wt[i, :len(seq)] = np.asarray([ALPHABET.index(aa) for aa in seq], dtype=np.int32)
        mask[i, :len(seq)] = 1
        residue_idx[i, :len(seq)] = np.arange(len(seq))
    for i, seq in enumerate(batch['aa_prefer']):
        S_prefer[i, :len(seq)] = np.asarray([ALPHABET.index(aa) for aa in seq], dtype=np.int32)
    for i, seq in enumerate(batch['aa_disprefer']):
        S_disprefer[i, :len(seq)] = np.asarray([ALPHABET.index(aa) for aa in seq], dtype=np.int32)
    S_prefer = torch.from_numpy(S_prefer).to(dtype=torch.long, device=device)
    S_disprefer = torch.from_numpy(S_disprefer).to(dtype=torch.long, device=device)
    S_wt = torch.from_numpy(S_wt).to(dtype=torch.long, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
    chain_M = mask.clone()
    chain_encoding_all = mask.clone()
    return X, S_wt, mask, chain_M, residue_idx, chain_encoding_all, S_prefer, S_disprefer

def featurize_test_mpnn(batch, device):
    B = batch['structure'].shape[0]
    L_max = max([len(x) for x in batch['aa_seq_wt']])
    X = batch['structure'][:, :L_max, :, :].to(dtype=torch.float32, device=device)
    S_wt = np.zeros([B, L_max], dtype=np.int32)
    mask = np.zeros([B, L_max], dtype=np.int32)
    residue_idx = -100*np.ones([B, L_max], dtype=np.int32)
    for i, seq in enumerate(batch['aa_seq_wt']):
        S_wt[i, :len(seq)] = np.asarray([ALPHABET.index(aa) for aa in seq], dtype=np.int32)
        mask[i, :len(seq)] = 1
        residue_idx[i, :len(seq)] = np.arange(len(seq))
    S_wt = torch.from_numpy(S_wt).to(dtype=torch.long, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
    chain_M = mask.clone()
    chain_encoding_all = mask.clone()    
    return X, S_wt, mask, chain_M, residue_idx, chain_encoding_all

def get_AbMask(batch, device):
    B = len(batch['aa_seq_wt'])
    L_max = max([len(x) for x in batch['aa_seq_wt']])
    mask = np.zeros([B, L_max], dtype=np.int32)
    Ab_mask = torch.zeros([B, L_max],dtype=torch.float32, device=device)
    for i, ab_lenth in enumerate(batch['Ab_lenth']):
        Ab_mask[i, :ab_lenth] = 1
        
    return Ab_mask

def mpnn_seq_to_tensor(aa_seq, mask, device):
    L_max = len(mask)
    S = np.zeros([L_max], dtype=np.int32)
    S[:len(aa_seq)] = np.asarray([ALPHABET.index(aa) for aa in aa_seq], dtype=np.int32)
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    return S


CLS_IDX=0
PADDING_idx=1
EOS_IDX=2
UNK_IDX=3
MASK_IDX=4

def featurize_esm(batch, esm_alphabet, device):
    batch_converter = esm_alphabet.get_batch_converter()
    
    # Get token & mask
    wt_data = [(pair_idx,wt_seq) for pair_idx,wt_seq in zip(batch['WT_name'],batch['aa_seq_wt'])]
    prefer_data = [(pair_idx,wt_seq) for pair_idx,wt_seq in zip(batch['WT_name'],batch['aa_prefer'])]
    disprefer_data = [(pair_idx,wt_seq) for pair_idx,wt_seq in zip(batch['WT_name'],batch['aa_disprefer'])]
    _,_,wt_batch_tokens=batch_converter(wt_data)
    _,_,prefer_batch_tokens=batch_converter(prefer_data)
    _,_,disprefer_batch_tokens=batch_converter(disprefer_data)
    mask = (wt_batch_tokens != PADDING_idx) & (wt_batch_tokens != EOS_IDX) & (wt_batch_tokens != CLS_IDX)
    
    # to device
    wt_batch_tokens = wt_batch_tokens.to(device=device)
    prefer_batch_tokens = prefer_batch_tokens.to(device=device)
    disprefer_batch_tokens = disprefer_batch_tokens.to(device=device)
    mask = mask.to(device=device)
    
    return wt_batch_tokens,prefer_batch_tokens,disprefer_batch_tokens, mask

def featurize_test_esm(batch, esm_alphabet, device):
    batch_converter = esm_alphabet.get_batch_converter()
    
    # Get token & mask
    wt_data = [(pair_idx,wt_seq) for pair_idx,wt_seq in zip(batch['WT_name'],batch['aa_seq_wt'])]
    _,_,wt_batch_tokens=batch_converter(wt_data)
    mask = (wt_batch_tokens != PADDING_idx) & (wt_batch_tokens != EOS_IDX) & (wt_batch_tokens != CLS_IDX)
    
    # to device
    wt_batch_tokens = wt_batch_tokens.to(device=device)
    mask = mask.to(device=device)
    
    return wt_batch_tokens, mask

def esm_seq_to_tensor(aa_seq, mask, esm_alphabet, device):
    batch_converter = esm_alphabet.get_batch_converter()
    # Get token & mask
    L_max = len(mask)
    S = torch.ones([L_max], dtype=torch.int64) # PADDING_idx=1 so use ones here
    data = [('p',aa_seq)]
    _,_,wt_batch_tokens=batch_converter(data)
    assert L_max>=len(wt_batch_tokens[0]), 'esm_seq_to_tensor error'
    S[:len(wt_batch_tokens[0])] = wt_batch_tokens[0]
    S = S.to(device=device)
    
    return S

def Saprot_combine_seq(residue_seq,struc_seq):
    assert len(residue_seq)==len(struc_seq),f'error, residue seq mismath struc seq {len(residue_seq)} {len(struc_seq)}'
    return "".join([aa.upper()+struc_aa.lower() for aa,struc_aa in zip(residue_seq,struc_seq)])

def featurize_saprot(batch, saprot_tokenizer, device):
    wt_seqs = [Saprot_combine_seq(wt_seq,struc_seq) for wt_seq,struc_seq in zip(batch['aa_seq_wt'],batch['struc_seq'])]
    prefer_seqs = [Saprot_combine_seq(wt_seq,struc_seq) for wt_seq,struc_seq in zip(batch['aa_prefer'],batch['struc_seq'])]
    disprefer_seqs = [Saprot_combine_seq(wt_seq,struc_seq) for wt_seq,struc_seq in zip(batch['aa_disprefer'],batch['struc_seq'])]
    
    wt_inputs = saprot_tokenizer(wt_seqs, return_tensors="pt",padding=True)
    wt_batch_tokens = wt_inputs['input_ids'].to(device=device)
    prefer_inputs = saprot_tokenizer(prefer_seqs, return_tensors="pt",padding=True)
    prefer_batch_tokens = prefer_inputs['input_ids'].to(device=device)
    disprefer_inputs = saprot_tokenizer(disprefer_seqs, return_tensors="pt",padding=True)
    disprefer_batch_tokens = disprefer_inputs['input_ids'].to(device=device)
    
    mask = (wt_batch_tokens != PADDING_idx) & (wt_batch_tokens != EOS_IDX) & (wt_batch_tokens != CLS_IDX)
    mask = mask.to(device=device)
    
    return wt_batch_tokens,prefer_batch_tokens,disprefer_batch_tokens, mask

def featurize_test_saprot(batch, saprot_tokenizer, device):   
    wt_seqs = [Saprot_combine_seq(wt_seq,struc_seq) for wt_seq,struc_seq in zip(batch['aa_seq_wt'],batch['struc_seq'])]
    
    wt_inputs = saprot_tokenizer(wt_seqs, return_tensors="pt",padding=True)
    wt_batch_tokens = wt_inputs['input_ids'].to(device=device)
        
    mask = (wt_batch_tokens != PADDING_idx) & (wt_batch_tokens != EOS_IDX) & (wt_batch_tokens != CLS_IDX)
    mask = mask.to(device=device)
    
    return wt_batch_tokens, mask

def saprot_seq_to_tensor(aa_seq,struc_seq, mask, saprot_tokenizer, device):
    L_max = len(mask)
    S = torch.ones([L_max], dtype=torch.int64)# PADDING_idx=1 so use ones here
    
    wt_seqs = [Saprot_combine_seq(aa_seq,struc_seq)]
    wt_inputs = saprot_tokenizer(wt_seqs, return_tensors="pt",padding=True)
    wt_batch_tokens = wt_inputs['input_ids']
    
    S[:len(wt_batch_tokens[0])] = wt_batch_tokens[0]
    S = S.to(device=device)
    return S



# other agents' logit to convert to mpnn format

ESM_TO_MPNN_MAPPING=[5, 23, 13, 9, 18, 6, 21, 12, 15, 4, 20, 17, 14, 16, 10, 8, 11, 7, 22, 19, 24]
SaProt_strucAA_TO_MPNN_MAPPING={
    'c':[24, 45, 66, 87, 108, 129, 150, 171, 192, 213, 234, 255, 276, 297, 318, 339, 360, 381, 402, 423, 3],
    'i':[22, 43, 64, 85, 106, 127, 148, 169, 190, 211, 232, 253, 274, 295, 316, 337, 358, 379, 400, 421, 3],
    't':[16, 37, 58, 79, 100, 121, 142, 163, 184, 205, 226, 247, 268, 289, 310, 331, 352, 373, 394, 415, 3],
    'y':[6, 27, 48, 69, 90, 111, 132, 153, 174, 195, 216, 237, 258, 279, 300, 321, 342, 363, 384, 405, 3],
    'w':[8, 29, 50, 71, 92, 113, 134, 155, 176, 197, 218, 239, 260, 281, 302, 323, 344, 365, 386, 407, 3],
    'n':[7, 28, 49, 70, 91, 112, 133, 154, 175, 196, 217, 238, 259, 280, 301, 322, 343, 364, 385, 406, 3],
    'm':[17, 38, 59, 80, 101, 122, 143, 164, 185, 206, 227, 248, 269, 290, 311, 332, 353, 374, 395, 416, 3],
    'd':[13, 34, 55, 76, 97, 118, 139, 160, 181, 202, 223, 244, 265, 286, 307, 328, 349, 370, 391, 412, 3],
    'r':[9, 30, 51, 72, 93, 114, 135, 156, 177, 198, 219, 240, 261, 282, 303, 324, 345, 366, 387, 408, 3],
    'a':[20, 41, 62, 83, 104, 125, 146, 167, 188, 209, 230, 251, 272, 293, 314, 335, 356, 377, 398, 419, 3],
    'f':[18, 39, 60, 81, 102, 123, 144, 165, 186, 207, 228, 249, 270, 291, 312, 333, 354, 375, 396, 417, 3],
    'l':[14, 35, 56, 77, 98, 119, 140, 161, 182, 203, 224, 245, 266, 287, 308, 329, 350, 371, 392, 413, 3],
    'k':[23, 44, 65, 86, 107, 128, 149, 170, 191, 212, 233, 254, 275, 296, 317, 338, 359, 380, 401, 422, 3],
    'v':[15, 36, 57, 78, 99, 120, 141, 162, 183, 204, 225, 246, 267, 288, 309, 330, 351, 372, 393, 414, 3],
    'e':[21, 42, 63, 84, 105, 126, 147, 168, 189, 210, 231, 252, 273, 294, 315, 336, 357, 378, 399, 420, 3],
    '#':[25, 46, 67, 88, 109, 130, 151, 172, 193, 214, 235, 256, 277, 298, 319, 340, 361, 382, 403, 424, 3],
    'p':[5, 26, 47, 68, 89, 110, 131, 152, 173, 194, 215, 236, 257, 278, 299, 320, 341, 362, 383, 404, 3],
    'g':[12, 33, 54, 75, 96, 117, 138, 159, 180, 201, 222, 243, 264, 285, 306, 327, 348, 369, 390, 411, 3],
    's':[19, 40, 61, 82, 103, 124, 145, 166, 187, 208, 229, 250, 271, 292, 313, 334, 355, 376, 397, 418, 3],
    'q':[10, 31, 52, 73, 94, 115, 136, 157, 178, 199, 220, 241, 262, 283, 304, 325, 346, 367, 388, 409, 3],
    'h':[11, 32, 53, 74, 95, 116, 137, 158, 179, 200, 221, 242, 263, 284, 305, 326, 347, 368, 389, 410, 3],
    'mask':[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
}
SaProt_idx_to_strucAA_MAPPING={
0:'mask',1:'mask',2:'mask',3:'mask',4:'mask',5:'p',6:'y',7:'n',8:'w',9:'r',10:'q',11:'h',12:'g',13:'d',14:'l',15:'v',16:'t',17:'m',18:'f',19:'s',20:'a',
21:'e',22:'i',23:'k',24:'c',25:'#',26:'p',27:'y',28:'n',29:'w',30:'r',31:'q',32:'h',33:'g',34:'d',35:'l',36:'v',37:'t',38:'m',39:'f',40:'s',41:'a',
42:'e',43:'i',44:'k',45:'c',46:'#',47:'p',48:'y',49:'n',50:'w',51:'r',52:'q',53:'h',54:'g',55:'d',56:'l',57:'v',58:'t',59:'m',60:'f',61:'s',62:'a',
63:'e',64:'i',65:'k',66:'c',67:'#',68:'p',69:'y',70:'n',71:'w',72:'r',73:'q',74:'h',75:'g',76:'d',77:'l',78:'v',79:'t',80:'m',81:'f',82:'s',83:'a',
84:'e',85:'i',86:'k',87:'c',88:'#',89:'p',90:'y',91:'n',92:'w',93:'r',94:'q',95:'h',96:'g',97:'d',98:'l',99:'v',100:'t',101:'m',102:'f',103:'s',104:'a',
105:'e',106:'i',107:'k',108:'c',109:'#',110:'p',111:'y',112:'n',113:'w',114:'r',115:'q',116:'h',117:'g',118:'d',119:'l',120:'v',121:'t',122:'m',123:'f',124:'s',125:'a',
126:'e',127:'i',128:'k',129:'c',130:'#',131:'p',132:'y',133:'n',134:'w',135:'r',136:'q',137:'h',138:'g',139:'d',140:'l',141:'v',142:'t',143:'m',144:'f',145:'s',146:'a',
147:'e',148:'i',149:'k',150:'c',151:'#',152:'p',153:'y',154:'n',155:'w',156:'r',157:'q',158:'h',159:'g',160:'d',161:'l',162:'v',163:'t',164:'m',165:'f',166:'s',167:'a',
168:'e',169:'i',170:'k',171:'c',172:'#',173:'p',174:'y',175:'n',176:'w',177:'r',178:'q',179:'h',180:'g',181:'d',182:'l',183:'v',184:'t',185:'m',186:'f',187:'s',188:'a',
189:'e',190:'i',191:'k',192:'c',193:'#',194:'p',195:'y',196:'n',197:'w',198:'r',199:'q',200:'h',201:'g',202:'d',203:'l',204:'v',205:'t',206:'m',207:'f',208:'s',209:'a',
210:'e',211:'i',212:'k',213:'c',214:'#',215:'p',216:'y',217:'n',218:'w',219:'r',220:'q',221:'h',222:'g',223:'d',224:'l',225:'v',226:'t',227:'m',228:'f',229:'s',230:'a',
231:'e',232:'i',233:'k',234:'c',235:'#',236:'p',237:'y',238:'n',239:'w',240:'r',241:'q',242:'h',243:'g',244:'d',245:'l',246:'v',247:'t',248:'m',249:'f',250:'s',251:'a',
252:'e',253:'i',254:'k',255:'c',256:'#',257:'p',258:'y',259:'n',260:'w',261:'r',262:'q',263:'h',264:'g',265:'d',266:'l',267:'v',268:'t',269:'m',270:'f',271:'s',272:'a',
273:'e',274:'i',275:'k',276:'c',277:'#',278:'p',279:'y',280:'n',281:'w',282:'r',283:'q',284:'h',285:'g',286:'d',287:'l',288:'v',289:'t',290:'m',291:'f',292:'s',293:'a',
294:'e',295:'i',296:'k',297:'c',298:'#',299:'p',300:'y',301:'n',302:'w',303:'r',304:'q',305:'h',306:'g',307:'d',308:'l',309:'v',310:'t',311:'m',312:'f',313:'s',314:'a',
315:'e',316:'i',317:'k',318:'c',319:'#',320:'p',321:'y',322:'n',323:'w',324:'r',325:'q',326:'h',327:'g',328:'d',329:'l',330:'v',331:'t',332:'m',333:'f',334:'s',335:'a',
336:'e',337:'i',338:'k',339:'c',340:'#',341:'p',342:'y',343:'n',344:'w',345:'r',346:'q',347:'h',348:'g',349:'d',350:'l',351:'v',352:'t',353:'m',354:'f',355:'s',356:'a',
357:'e',358:'i',359:'k',360:'c',361:'#',362:'p',363:'y',364:'n',365:'w',366:'r',367:'q',368:'h',369:'g',370:'d',371:'l',372:'v',373:'t',374:'m',375:'f',376:'s',377:'a',
378:'e',379:'i',380:'k',381:'c',382:'#',383:'p',384:'y',385:'n',386:'w',387:'r',388:'q',389:'h',390:'g',391:'d',392:'l',393:'v',394:'t',395:'m',396:'f',397:'s',398:'a',
399:'e',400:'i',401:'k',402:'c',403:'#',404:'p',405:'y',406:'n',407:'w',408:'r',409:'q',410:'h',411:'g',412:'d',413:'l',414:'v',415:'t',416:'m',417:'f',418:'s',419:'a',
420:'e',421:'i',422:'k',423:'c',424:'#',425:'p',426:'y',427:'n',428:'w',429:'r',430:'q',431:'h',432:'g',433:'d',434:'l',435:'v',436:'t',437:'m',438:'f',439:'s',440:'a',
441:'e',442:'i',443:'k',444:'c',445:'#',
}
SaProt_idx_TO_MPNN_MAPPING={idx:SaProt_strucAA_TO_MPNN_MAPPING[struc_aa] for idx,struc_aa in SaProt_idx_to_strucAA_MAPPING.items()}
SaProt_idx_TO_MPNN_MAPPING_MATRIX=[SaProt_idx_TO_MPNN_MAPPING[i] for i in range(len(SaProt_idx_TO_MPNN_MAPPING))]
SaProt_idx_TO_MPNN_MAPPING_MATRIX = torch.tensor(SaProt_idx_TO_MPNN_MAPPING_MATRIX)

def map_to_mpnn_logits(origin_logits, origin_mask, mpnn_mask,from_model,batch_tokens):
    if from_model=='esm':
        origin_logits_mapped = origin_logits[:, :, ESM_TO_MPNN_MAPPING]  # [B, L', 21]
    elif from_model=='saprot':
        Mapping = SaProt_idx_TO_MPNN_MAPPING_MATRIX.to(batch_tokens.device)[batch_tokens]
        Mapping = Mapping.to(origin_logits.device)
        origin_logits_mapped = torch.gather(origin_logits, dim=-1, index=Mapping) # [B,L',21]
                
    B, L = mpnn_mask.shape
    converted_logits = torch.zeros((B, L, len(ESM_TO_MPNN_MAPPING)), device=origin_logits.device)
    
    for b in range(B):
        # Get the valid index of each sample
        origin_valid_logits = origin_logits_mapped[b][origin_mask[b]]
        mpnn_valid_idx = torch.nonzero(mpnn_mask[b], as_tuple=False).squeeze(-1)  # Valid position index of ProteinMPNN
        assert origin_valid_logits.shape[0] == mpnn_valid_idx.shape[0], f"Mask mismatch at batch {b}: origin_valid_idx={origin_valid_logits.shape[0]}, mpnn_valid_idx={mpnn_valid_idx.shape[0]}"
        converted_logits[b, mpnn_valid_idx, :] = origin_valid_logits  # Mapping to the position of ProteinMPNN

    return converted_logits

def softmax_multi_agent_logits(logits_tuple,mask_tuple,saprot_batch_tokens):
    mpnn_logits,esm_logits,saprot_logits = logits_tuple
    mpnn_mask,esm_mask,saprot_mask = mask_tuple
    esm_to_mpnn_logits = map_to_mpnn_logits(esm_logits,esm_mask,mpnn_mask,'esm',None)
    assert mpnn_logits.shape == esm_to_mpnn_logits.shape, "esm logits mismatch shape"
    saprot_to_mpnn_logits = map_to_mpnn_logits(saprot_logits,saprot_mask,mpnn_mask,'saprot',saprot_batch_tokens)
    assert mpnn_logits.shape == saprot_to_mpnn_logits.shape, "sarpto logits mismatch shape"
    
    mpnn_log_probs = torch.log_softmax(mpnn_logits, dim=-1)
    esm_to_mpnn_log_probs = torch.log_softmax(esm_to_mpnn_logits, dim=-1)
    saprot_to_mpnn_log_probs = torch.log_softmax(saprot_to_mpnn_logits, dim=-1)
    Multi_agent_log_probs = (mpnn_log_probs + esm_to_mpnn_log_probs + saprot_to_mpnn_log_probs) / 3.0
    
    return Multi_agent_log_probs






# train test process batch
def MultiAgent_process_batch(model_policy,model_ref, batch, loss_func, device, stage='train', optimizer=None, is_AbDesign=False):
    batch = recursive_to(batch, device)
    if stage=='train' or stage=='iteration_train':
        model_policy.train()
        X, mpnn_S_wt, mpnn_mask, chain_M, residue_idx, chain_encoding_all, mpnn_S_prefer, mpnn_S_disprefer = featurize_mpnn(batch,device)
        esm_S_wt,esm_S_prefer,esm_S_disprefer, esm_mask = featurize_esm(batch,model_policy.esm_alphabet,device)
        saprot_S_wt,saprot_S_prefer,saprot_S_disprefer, saprot_mask = featurize_saprot(batch,model_policy.saprot_tokenizer,device)

        log_probs_policy,logits_policy = model_policy(esm_S_wt, saprot_S_wt, mpnn_S_wt, mpnn_mask, X, chain_M, residue_idx, chain_encoding_all)
        Multi_agent_log_probs_policy = softmax_multi_agent_logits(logits_policy,(mpnn_mask,esm_mask,saprot_mask),saprot_S_wt)
        
        model_ref.eval()
        with torch.no_grad():
            log_probs_ref,logits_ref = model_ref(esm_S_wt, saprot_S_wt, mpnn_S_wt, mpnn_mask, X, chain_M, residue_idx, chain_encoding_all)
            Multi_agent_log_probs_ref = softmax_multi_agent_logits(logits_ref,(mpnn_mask,esm_mask,saprot_mask),saprot_S_wt)
        
        log_probs_policy = (log_probs_policy[0],log_probs_policy[1],log_probs_policy[2],Multi_agent_log_probs_policy)
        log_probs_ref = (log_probs_ref[0],log_probs_ref[1],log_probs_ref[2],Multi_agent_log_probs_ref)
        each_agent_encoding_pair_list=[(mpnn_S_prefer,mpnn_S_disprefer,mpnn_mask),(esm_S_prefer,esm_S_disprefer, esm_mask),(saprot_S_prefer,saprot_S_disprefer, saprot_mask),(mpnn_S_prefer,mpnn_S_disprefer,mpnn_mask),]
        noise=None
        if stage=='iteration_train':
            noise=batch['noise']
        loss = loss_func(each_agent_encoding_pair_list,log_probs_policy,log_probs_ref,noise)
        
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_policy.parameters(), 100.0)
        optimizer.step()
        optimizer.zero_grad()
        return loss
    elif stage=='pareto':
        model_policy.eval()
        X, mpnn_S_wt, mpnn_mask, chain_M, residue_idx, chain_encoding_all = featurize_test_mpnn(batch, device)
        esm_S_wt, esm_mask = featurize_test_esm(batch,model_policy.esm_alphabet,device)
        saprot_S_wt, saprot_mask = featurize_test_saprot(batch,model_policy.saprot_tokenizer,device)
        
        with torch.no_grad():
            log_probs_policy,_ = model_policy(esm_S_wt, saprot_S_wt, mpnn_S_wt, mpnn_mask, X, chain_M, residue_idx, chain_encoding_all)
        return log_probs_policy, (mpnn_mask,esm_mask,saprot_mask)
        
    elif stage=='logit':
        model_policy.eval()
        X, mpnn_S_wt, mpnn_mask, chain_M, residue_idx, chain_encoding_all = featurize_test_mpnn(batch, device)
        esm_S_wt, esm_mask = featurize_test_esm(batch,model_policy.esm_alphabet,device)
        saprot_S_wt, saprot_mask = featurize_test_saprot(batch,model_policy.saprot_tokenizer,device)
        
        ab_mask=None
        with torch.no_grad():
            log_probs_policy,logits_policy = model_policy(esm_S_wt, saprot_S_wt, mpnn_S_wt, mpnn_mask, X, chain_M, residue_idx, chain_encoding_all)
            
            esm_to_mpnn_logits = map_to_mpnn_logits(logits_policy[1],esm_mask,mpnn_mask,'esm',None)
            saprot_to_mpnn_logits = map_to_mpnn_logits(logits_policy[2],saprot_mask,mpnn_mask,'saprot',saprot_S_wt)
        if is_AbDesign:
            ab_mask = get_AbMask(batch,device)
        return ab_mask, (esm_to_mpnn_logits,saprot_to_mpnn_logits), (mpnn_mask,esm_mask,saprot_mask)
    
def deepcopy_modelmanager(from_modelmanager,to_modelmanager):
    sate_dict = from_modelmanager.state_dict()
    to_modelmanager.load_state_dict(sate_dict)
    return to_modelmanager

def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        try:
            return obj.cuda(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device=device) for k, v in obj.items()}

    else:
        return obj

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def check_dir(path, overwrite=True):
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        pass

def write_and_print(log,msg):
    print(f"\033[0;30;46m{msg}\033[0m")
    log.write(f"{msg}\n")

def Metric(y_true,y_pred):
    pcc = round(pearsonr(y_true, y_pred)[0],4)
    spc = round(spearmanr(y_true,y_pred)[0],4)
    bin_y_true = [1 if i > 0 else 0 for i in y_true]
    if sum(bin_y_true)==0 or sum(bin_y_true)==len(bin_y_true):
        auc = -1e9
    else:
        auc = roc_auc_score(bin_y_true, y_pred)
    return pcc,spc,auc



def read_fasta(filepath):
    sequences = {}
    with open(filepath, 'r') as f:
        seq_id = None
        seq_chunks = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if seq_id:
                    sequences[seq_id] = ''.join(seq_chunks)
                seq_id = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if seq_id:
            sequences[seq_id] = ''.join(seq_chunks)
    return sequences



    
def cal_rmsd(S_sp, S, batch, the_folding_model, pdb_path, mask_for_loss, save_path):
    with torch.no_grad():      
        results_list = []  
        sc_output_dir_base = os.path.join(save_path, 'sc_eval','sc_output', batch["WT_name"][0][:-4])
        sc_output_dir = os.path.join(sc_output_dir_base, 'true')
        the_pdb_path = os.path.join(pdb_path, batch['WT_name'][0])
        # fold the ground truth sequence
        os.makedirs(os.path.join(sc_output_dir, 'true_seqs'), exist_ok=True)
        true_fasta = fasta.FastaFile()
        true_detok_seq = "".join([ALPHABET[x] for _ix, x in enumerate(S[0]) if mask_for_loss[0][_ix] == 1])
        true_fasta['true_seq_1'] = true_detok_seq
        true_fasta_path = os.path.join(sc_output_dir, 'true_seqs', 'true.fa')
        true_fasta.write(true_fasta_path)
        
        true_folded_dir = os.path.join(sc_output_dir, 'true_folded')
        true_folded_pdb_path = os.path.join(true_folded_dir, 'folded_true_seq_1.pdb')
        if not os.path.exists(true_folded_pdb_path):
            if os.path.exists(true_folded_dir):
                shutil.rmtree(true_folded_dir)
            os.makedirs(true_folded_dir, exist_ok=False)
            true_folded_output = the_folding_model.fold_fasta(true_fasta_path, true_folded_dir)
        
        true_folded_pose = pyrosetta.pose_from_file(true_folded_pdb_path)

        scorefxn = pyrosetta.create_score_function("ref2015_cart")
        
        tf = TaskFactory()
        tf.push_back(RestrictToRepacking())
        packer = PackRotamersMover(scorefxn, tf.create_task_and_apply_taskoperations(true_folded_pose))
        packer.apply(true_folded_pose)
        relax = FastRelax()
        relax.set_scorefxn(scorefxn)
        relax.apply(true_folded_pose)
        true_folded_relax_path = os.path.join(sc_output_dir, 'true_folded_relax', 'folded_true_relax_seq_1.pdb')
        os.makedirs(os.path.join(sc_output_dir, 'true_folded_relax'), exist_ok=True)
        true_folded_pose.dump_pdb(true_folded_relax_path)

        true_pose = pyrosetta.pose_from_file(the_pdb_path)
        tf = TaskFactory()
        tf.push_back(RestrictToRepacking())
        packer = PackRotamersMover(scorefxn, tf.create_task_and_apply_taskoperations(true_pose))
        packer.apply(true_pose)
        relax = FastRelax()
        relax.set_scorefxn(scorefxn)
        relax.apply(true_pose)

        os.makedirs(os.path.join(sc_output_dir, 'true_relax'), exist_ok=True)
        true_pose.dump_pdb(os.path.join(sc_output_dir, 'true_relax', 'true_relax_seq_1.pdb'))

        foldtrue_true_bbrmsd = pyrosetta.rosetta.core.scoring.bb_rmsd(true_pose, true_folded_pose)
        print("start each generated seq")
        
        
        for _it, ssp in tqdm(enumerate(S_sp)):
            num = _it
            sc_output_dir = os.path.join(sc_output_dir_base, f'{num}')
            os.makedirs(sc_output_dir, exist_ok=True)
            os.makedirs(os.path.join(sc_output_dir, 'fmif_seqs'), exist_ok=True)
            codesign_fasta = fasta.FastaFile()
            detok_seq = "".join([ALPHABET[x] for _ix, x in enumerate(ssp) if mask_for_loss[_it][_ix] == 1])
            codesign_fasta['codesign_seq_1'] = detok_seq
            codesign_fasta_path = os.path.join(sc_output_dir, 'fmif_seqs', 'codesign.fa')
            codesign_fasta.write(codesign_fasta_path)

            
            folded_dir = os.path.join(sc_output_dir, 'folded')
            if os.path.exists(folded_dir):
                shutil.rmtree(folded_dir)
            os.makedirs(folded_dir, exist_ok=False)
            gen_folded_pdb_path = os.path.join(folded_dir, 'folded_codesign_seq_1.pdb')
            
            folded_output = the_folding_model.fold_fasta(codesign_fasta_path, folded_dir)
            torch.cuda.empty_cache()

            
            pose = pyrosetta.pose_from_file(gen_folded_pdb_path)

            tf = TaskFactory()
            tf.push_back(RestrictToRepacking())
            packer = PackRotamersMover(scorefxn, tf.create_task_and_apply_taskoperations(pose))
            packer.apply(pose)

            relax = FastRelax()
            relax.set_scorefxn(scorefxn)
            relax.apply(pose)

            gen_true_bbrmsd = pyrosetta.rosetta.core.scoring.bb_rmsd(true_pose, pose)
            gen_foldtrue_bbrmsd = pyrosetta.rosetta.core.scoring.bb_rmsd(true_folded_pose, pose)
            seq_revovery = (S_sp[_it] == S[0]).float().mean().item()

            resultdf = pd.DataFrame(columns=['gen_true_bb_rmsd', 'gen_foldtrue_bb_rmsd', 'foldtrue_true_bb_rmsd', 'seq_recovery'])
            resultdf.loc[0] = [gen_true_bbrmsd, gen_foldtrue_bbrmsd, foldtrue_true_bbrmsd, seq_revovery]
            resultdf['seq'] = "".join([ALPHABET[x] for _ix, x in enumerate(ssp) if mask_for_loss[_it][_ix] == 1])
            resultdf['true_seq'] = true_detok_seq
            resultdf['WT_name'] = batch['WT_name'][0]
            resultdf['num'] = num
            resultdf['pdbpath'] = sc_output_dir
            results_list.append(resultdf)

    return results_list


def MultiAgent_sample_sequence(config,model_predictor,esm_logits,saprot_logits,ab_mask=None):

    import time, os
    import numpy as np
    import torch
    import copy
    import random
    import os.path
    import subprocess
    
    from common_utils.protein_mpnn_utils import _S_to_seq, tied_featurize, parse_PDB
    from common_utils.protein_mpnn_utils import StructureDatasetPDB, ProteinMPNN

    seed=config['sample seed']
    folder_for_outputs = config['ouput_path']
    num_seq_per_target = config['num_seq_per_target']
    sampling_temp = config['sampling_temp']
    pdb_path = config['pdb_path']
    batch_size = config['sample_batchsize']
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)   
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    
    MAX_LENTH = 20000
    print_all =True

    
    model = model_predictor.mpnn
    model.eval()
    
    NUM_BATCHES = num_seq_per_target//batch_size
    BATCH_COPIES = batch_size
    temperatures = [float(item) for item in sampling_temp.split()]
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    omit_AAs_np = np.array([AA in ['X'] for AA in alphabet]).astype(np.float32)
    
    bias_AAs_np = np.zeros(len(alphabet))
    
    
    pdb_dict_list = parse_PDB(pdb_path)
    dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=MAX_LENTH)
    all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9]=='seq_chain'] #['A','B', 'C',...]
    designed_chain_list = all_chain_list
    chain_id_dict = {}
    chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, [])

 
    # Build paths for experiment
    base_folder = folder_for_outputs
    if base_folder[-1] != '/':
        base_folder = base_folder + '/'
    os.makedirs(base_folder,exist_ok=True)
    os.makedirs(base_folder + 'seqs',exist_ok=True)
    os.makedirs(base_folder + 'scores',exist_ok=True)
    os.makedirs(base_folder + 'probs',exist_ok=True) 
    
    # Timing
    # Validation epoch
    with torch.no_grad():
        for ix, protein in enumerate(dataset_valid):
            score_list = []
            global_score_list = []
            all_probs_list = []
            all_log_probs_list = []
            S_sample_list = []
            batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
            X, S, mask, _, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, _, _, _, _, _, bias_by_res_all, _ = tied_featurize(batch_clones, device, chain_id_dict)
            if ab_mask != None:
                mask=ab_mask.expand_as(mask)
            
            name_ = batch_clones[0]['name']

            randn_1 = torch.randn(chain_M.shape, device=X.device)
            log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
            mask_for_loss = mask*chain_M*chain_M_pos
            scores = _scores(S, log_probs, mask_for_loss) #score only the redesigned part
            native_score = scores.cpu().data.numpy()
            global_scores = _scores(S, log_probs, mask) #score the whole structure-sequence
            global_native_score = global_scores.cpu().data.numpy()
            # Generate some sequences
            ali_file = base_folder + '/seqs/' + batch_clones[0]['name'] + '.fa'
            if print_all:
                print(f'Generating sequences for: {name_}')
            t0 = time.time()
            with open(ali_file, 'w') as f:
                for temp in temperatures:
                    for j in range(NUM_BATCHES):
                        randn_2 = torch.randn(chain_M.shape, device=X.device)
                        sample_dict = model.simplified_sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, esm_logits, saprot_logits, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, bias_by_res=bias_by_res_all)
                        
                        S_sample = sample_dict["S"] 
                        log_probs = model(X, S_sample, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2, use_input_decoding_order=True, decoding_order=sample_dict["decoding_order"])
                        mask_for_loss = mask*chain_M*chain_M_pos
                        scores = _scores(S_sample, log_probs, mask_for_loss)
                        scores = scores.cpu().data.numpy()
                        
                        global_scores = _scores(S_sample, log_probs, mask) #score the whole structure-sequence
                        global_scores = global_scores.cpu().data.numpy()
                        
                        all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                        all_log_probs_list.append(log_probs.cpu().data.numpy())
                        S_sample_list.append(S_sample.cpu().data.numpy())
                        for b_ix in range(BATCH_COPIES):
                            masked_chain_length_list = masked_chain_length_list_list[b_ix]
                            masked_list = masked_list_list[b_ix]
                            seq_recovery_rate = torch.sum(torch.sum(torch.nn.functional.one_hot(S[b_ix], 21)*torch.nn.functional.one_hot(S_sample[b_ix], 21),axis=-1)*mask_for_loss[b_ix])/torch.sum(mask_for_loss[b_ix])
                            seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                            score = scores[b_ix]
                            score_list.append(score)
                            global_score = global_scores[b_ix]
                            global_score_list.append(global_score)
                            native_seq = _S_to_seq(S[b_ix], chain_M[b_ix])
                            if b_ix == 0 and j==0 and temp==temperatures[0]:
                                start = 0
                                end = 0
                                list_of_AAs = []
                                for mask_l in masked_chain_length_list:
                                    end += mask_l
                                    list_of_AAs.append(native_seq[start:end])
                                    start = end
                                native_seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                                l0 = 0
                                for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                                    l0 += mc_length
                                    native_seq = native_seq[:l0] + '/' + native_seq[l0:]
                                    l0 += 1
                                sorted_masked_chain_letters = np.argsort(masked_list_list[0])
                                print_masked_chains = [masked_list_list[0][i] for i in sorted_masked_chain_letters]
                                sorted_visible_chain_letters = np.argsort(visible_list_list[0])
                                print_visible_chains = [visible_list_list[0][i] for i in sorted_visible_chain_letters]
                                native_score_print = np.format_float_positional(np.float32(native_score.mean()), unique=False, precision=4)
                                global_native_score_print = np.format_float_positional(np.float32(global_native_score.mean()), unique=False, precision=4)
                                script_dir = os.path.dirname(os.path.realpath(__file__))
                                try:
                                    commit_str = subprocess.check_output(f'git --git-dir {script_dir}/.git rev-parse HEAD', shell=True, stderr=subprocess.DEVNULL).decode().strip()
                                except subprocess.CalledProcessError:
                                    commit_str = 'unknown'
                                f.write('>{}, score={}, global_score={}, fixed_chains={}, designed_chains={}, git_hash={}, seed={}\n{}\n'.format(name_, native_score_print, global_native_score_print, print_visible_chains, print_masked_chains, commit_str, seed, native_seq)) #write the native sequence
                            start = 0
                            end = 0
                            list_of_AAs = []
                            for mask_l in masked_chain_length_list:
                                end += mask_l
                                list_of_AAs.append(seq[start:end])
                                start = end

                            seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                            l0 = 0
                            for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                                l0 += mc_length
                                seq = seq[:l0] + '/' + seq[l0:]
                                l0 += 1
                            score_print = np.format_float_positional(np.float32(score), unique=False, precision=4)
                            global_score_print = np.format_float_positional(np.float32(global_score), unique=False, precision=4)
                            seq_rec_print = np.format_float_positional(np.float32(seq_recovery_rate.detach().cpu().numpy()), unique=False, precision=4)
                            sample_number = j*BATCH_COPIES+b_ix+1
                            f.write('>T={}, sample={}, score={}, global_score={}, seq_recovery={}\n{}\n'.format(temp,sample_number,score_print,global_score_print,seq_rec_print,seq)) #write generated sequence
            

            t1 = time.time()
            dt = round(float(t1-t0), 4)
            num_seqs = len(temperatures)*NUM_BATCHES*BATCH_COPIES
            total_length = X.shape[1]
            if print_all:
                print(f'{num_seqs} sequences of length {total_length} generated in {dt} seconds')