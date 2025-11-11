from Levenshtein import distance as levenshtein
import numpy as np
import torch
import os
import pandas as pd
from omegaconf import OmegaConf
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


to_np = lambda x: x.cpu().detach().numpy()
to_list = lambda x: to_np(x).tolist()
alphabet = "ARNDCQEGHILKMFPSTWYV"

class LengthMaxPool1D(nn.Module):
    def __init__(self, in_dim, out_dim, linear=False, activation='relu'):
        super().__init__()
        self.linear = linear
        if self.linear:
            self.layer = nn.Linear(in_dim, out_dim)

        if activation == 'swish':
            self.act_fn = lambda x: x * torch.sigmoid(100.0*x)
        elif activation == 'softplus':
            self.act_fn = nn.Softplus()
        elif activation == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        elif activation == 'leakyrelu':
            self.act_fn = nn.LeakyReLU()
        elif activation == 'relu':
            self.act_fn = lambda x: F.relu(x)
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.linear:
            x = self.act_fn(self.layer(x))
        x = torch.max(x, dim=1)[0]
        return x


class BaseCNN(nn.Module):
    def __init__(
            self,
            n_tokens: int = 20,
            kernel_size: int = 5 ,
            input_size: int = 256,
            dropout: float = 0.0,
            make_one_hot=True,
            activation: str = 'relu',
            linear: bool=True,
            **kwargs):
        super(BaseCNN, self).__init__()
        self.encoder = nn.Conv1d(n_tokens, input_size, kernel_size=kernel_size)
        self.embedding = LengthMaxPool1D(
            linear=linear,
            in_dim=input_size,
            out_dim=input_size*2,
            activation=activation,
        )
        self.decoder = nn.Linear(input_size*2, 1)
        self.n_tokens = n_tokens
        self.dropout = nn.Dropout(dropout) # TODO: actually add this to model
        self.input_size = input_size
        self._make_one_hot = make_one_hot

    def forward(self, x):
        #onehotize
        if self._make_one_hot:
            x = F.one_hot(x.long(), num_classes=self.n_tokens)
        x = x.permute(0, 2, 1).float()
        # encoder
        x = self.encoder(x).permute(0, 2, 1)
        x = self.dropout(x)
        # embed
        x = self.embedding(x)
        # decoder
        output = self.decoder(x).squeeze(1)
        return output

class Encoder(object):
    """convert between strings and their one-hot representations"""
    def __init__(self, alphabet: str = 'ARNDCQEGHILKMFPSTWYV'):
        self.alphabet = alphabet
        self.a_to_t = {a: i for i, a in enumerate(self.alphabet)}
        self.t_to_a = {i: a for i, a in enumerate(self.alphabet)}

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet)
    
    @property
    def vocab(self) -> np.ndarray:
        return np.array(list(self.alphabet))
    
    @property
    def tokenized_vocab(self) -> np.ndarray:
        return np.array([self.a_to_t[a] for a in self.alphabet])

    def onehotize(self, batch):
        #create a tensor, and then onehotize using scatter_
        onehot = torch.zeros(len(batch), self.vocab_size)
        onehot.scatter_(1, batch.unsqueeze(1), 1)
        return onehot
    
    def encode(self, seq_or_batch: str or list, return_tensor = True) -> np.ndarray or torch.Tensor:
        if isinstance(seq_or_batch, str):
            encoded_list = [self.a_to_t[a] for a in seq_or_batch]
        else:
            encoded_list = [[self.a_to_t[a] for a in seq] for seq in seq_or_batch]
        return torch.tensor(encoded_list) if return_tensor else encoded_list
    
    def decode(self, x: np.ndarray or list or torch.Tensor) -> str or list:
        if isinstance(x, np.ndarray):
            x = x.tolist()
        elif isinstance(x, torch.Tensor):
            x = x.tolist()

        if isinstance(x[0], list):
            return [''.join([self.t_to_a[t] for t in xi]) for xi in x]
        else:
            return ''.join([self.t_to_a[t] for t in x])

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

def compute_distance100(generated_seqs, ground_truth_seqs, verbose=False):
    assert len(generated_seqs) == 100 and len(ground_truth_seqs) == 100, "需要传入100个生成序列"
    
    min_distances = []
    for i, gen_seq in enumerate(generated_seqs):
        min_distance = min([levenshtein(gen_seq, gt_seq) for gt_seq in ground_truth_seqs])
        min_distances.append(min_distance)
    return min_distances
    
class EvalRunner:
    def __init__(self, runner_cfg,device):
        # tokenizer
        self.runner_cfg = runner_cfg
        self.device = device
        self.predictor_tokenizer = Encoder()
                
        gt_csv = pd.read_csv(runner_cfg['gt_csv_path'])
        self._max_known_score = np.max(gt_csv.score)
        self._min_known_score = np.min(gt_csv.score)
        self.normalize = lambda x: to_np((x - self._min_known_score) / (self._max_known_score - self._min_known_score)).item()
        
        gt_csv_sorted = gt_csv.sort_values(by="score", ascending=False)
        self.top100_sequences = gt_csv_sorted.head(100).sequence.tolist()
                        
        # evaluator
        oracle_dir = runner_cfg['oracle_dir']
        cfg_path = os.path.join(oracle_dir, 'config.yaml')
        oracle_path = os.path.join(oracle_dir, 'cnn_oracle.ckpt')
        oracle_state_dict = torch.load(oracle_path, map_location=self.device)
        with open(cfg_path, 'r') as fp:
            ckpt_cfg = OmegaConf.load(fp.name)
        self._cnn_oracle = BaseCNN(**ckpt_cfg.model.predictor) #oracle has same architecture as predictor
        self._cnn_oracle.load_state_dict(
            {k.replace('predictor.', ''): v for k,v in oracle_state_dict['state_dict'].items()})
        self._cnn_oracle = self._cnn_oracle.to(self.device)
        self._cnn_oracle.eval()
        self.run_oracle = self._run_cnn_oracle
        
        import pickle
        self._base_pool_seqs = [data_dict[0] for _,data_dict in pickle.load(open(runner_cfg['train_data_path'],'rb')).items()]
    
    def novelty(self, sampled_seqs):
        all_novelty = []
        for src in sampled_seqs:  
            min_dist = 1e9
            for known in self._base_pool_seqs:
                dist = levenshtein(src, known)
                if dist < min_dist:
                    min_dist = dist
            all_novelty.append(min_dist)
        return all_novelty
    
    def tokenize(self, seqs):
        return self.predictor_tokenizer.encode(seqs).to(self.device)

    def _run_cnn_oracle(self, seqs):
        tokenized_seqs = self.tokenize(seqs)
        batches = torch.split(tokenized_seqs, self.runner_cfg['batch_size'], 0)
        scores = []
        for b in batches:
            if b is None:
                continue
            results = self._cnn_oracle(b).detach()
            scores.append(results)
        return torch.concat(scores, dim=0)
    
    def evaluate_sequences(self, generated_seqs):
        generated_seqs_Deduplication=set()
        for seq in generated_seqs:
            generated_seqs_Deduplication.add(seq)
            if len(generated_seqs_Deduplication)>=128:
                break
        generated_seqs = list(generated_seqs_Deduplication)
            
        seq_novelty = self.novelty(generated_seqs)
        generated_seqs_score = self.run_oracle(generated_seqs)
        normalized_scores = [self.normalize(x) for x in generated_seqs_score]
        generated_seqs_score = to_np(generated_seqs_score)
        results_df = pd.DataFrame({
            'sequence': generated_seqs,
            'oracle_score': generated_seqs_score,
            'normalized_score': normalized_scores,
        })

        
        seq_diversity = diversity(generated_seqs)

        metrics_dict = {
            'fitness_median': np.median(normalized_scores),
            'fitness_mean': np.mean(normalized_scores),
            'fitness_max': np.max(normalized_scores),
            
            'diversity_median': np.median(seq_diversity),
            'diversity_mean': np.mean(seq_diversity),
            
            'novelty_mean': np.mean(seq_novelty),
            'novelty_median': np.median(seq_novelty),
            
            'num_unique': len(set(generated_seqs)),
        }
        return results_df, metrics_dict
    