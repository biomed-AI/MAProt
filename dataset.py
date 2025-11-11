import pickle
from Bio.PDB import PDBParser
import numpy as np
import torch
from torch.utils.data import DataLoader

import os
from torch.utils.data import Dataset, DataLoader
from common_utils.foldseek_util import get_struc_seq


def get_dataloader(args,split,data_path):
    pdb_path = args.pdb_path
    dpo_data_dict = pickle.load(open(data_path, 'rb'))
    dpo_wt_dataset = ProteinDPOTestDataset(dpo_data_dict, args) 
    loader_wt = DataLoader(dpo_wt_dataset, batch_size=1, shuffle=False) 
    
    if split == 'train':
        dpo_pair_dataset = ProteinDPOTrainDataset(dpo_data_dict,args)
        loader_train_dpo_pair = DataLoader(dpo_pair_dataset, batch_size=args.batch_size, shuffle=True)            
        wt_name_struc_seq_dict = dpo_wt_dataset.get_wt_name_struc_seq_dict()
        return loader_train_dpo_pair,loader_wt,wt_name_struc_seq_dict
    
    elif split == 'test':
        return loader_wt
    
class ProteinStructureDataset(Dataset):
    def __init__(self, directory, is_AbDesign, max_len):
        self.directory = directory
        self.max_len = max_len
        self.filenames = [f for f in os.listdir(directory) if f.endswith('.pdb')]
        self.parser = PDBParser(QUIET=True)
        self.is_AbDesign = is_AbDesign

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.filenames[idx])
        structure = self.parser.get_structure(id=None, file=file_path)
        model = structure[0]  # Assuming only one model per PDB file

        # Extract coordinates for N, CA, C, O atoms for each residue
        coords = []
        for residue in model.get_residues():
            try:
                n = residue['N'].get_coord()
                ca = residue['CA'].get_coord()
                c = residue['C'].get_coord()
                o = residue['O'].get_coord()
                coords.append([n, ca, c, o])
            except KeyError:
                continue  # Skip residues that do not have all atoms

        # Pad the coordinates to max_len
        coords = np.array(coords, dtype=np.float32)
        if len(coords) < self.max_len:
            padding = np.zeros((self.max_len - len(coords), 4, 3), dtype=np.float32)
            coords = np.concatenate((coords, padding), axis=0)
        elif len(coords) > self.max_len:
            # print(f"Protein sequence in {self.filenames[idx]} is {len(coords)} longer than max_len {self.max_len}. Truncating.")
            coords = coords[:self.max_len]

        # SaProt seq
        if self.is_AbDesign:
            parsed_seqs_Ab = get_struc_seq("bin/foldseek", file_path, ["A"], plddt_mask=False)["A"]
            parsed_seqs_Ag = get_struc_seq("bin/foldseek", file_path, ["B"], plddt_mask=False)["B"]
            _, foldseek_seq_Ab, _ = parsed_seqs_Ab
            _, foldseek_seq_Ag, _ = parsed_seqs_Ag
            foldseek_seq = foldseek_seq_Ab.lower() + foldseek_seq_Ag.lower()
        else:    
            parsed_seqs = get_struc_seq("bin/foldseek", file_path, ["A"], plddt_mask=False)["A"]
            seq, foldseek_seq, combined_seq = parsed_seqs
            foldseek_seq = foldseek_seq.lower()
        
        return torch.tensor(coords), self.filenames[idx], foldseek_seq


class ProteinDPOTestDataset(Dataset):
    def __init__(self, dpo_test_dict,args):
        self.is_AbDesign = args.dataset=='AbDesign'

        max_lenth = max([len(data[0]+data[6])+10 for _,data in dpo_test_dict.items()]) if self.is_AbDesign else max([len(data[0])+10 for _,data in dpo_test_dict.items()])
        dataset = ProteinStructureDataset(args.pdb_path, self.is_AbDesign, max_lenth) 
        loader = DataLoader(dataset, batch_size=10000, shuffle=False)
        for batch in loader:
            pdb_structures = batch[0]
            pdb_filenames = batch[1]
            SaProt_foldseek_seq = batch[2]
            pdb_idx_dict = {pdb_filenames[i]: i for i in range(len(pdb_filenames))}
            break
        self.pdb_idx_dict = pdb_idx_dict
        self.pdb_structure = pdb_structures
        self.SaProt_foldseek_seq = SaProt_foldseek_seq
        
        
        self.dpo_test_dict = dpo_test_dict
        self.WT_name_data_dict = self.process_wt_data(self.dpo_test_dict)
        self.WT_names = list(self.WT_name_data_dict.keys())

    def get_wt_name_struc_seq_dict(self):
        wt_name_struc_seq_dict = {wt_name:self.SaProt_foldseek_seq[idx] for wt_name,idx in self.pdb_idx_dict.items()}
        return wt_name_struc_seq_dict
    
    def process_wt_data(self,dpo_test_dict):
        WT_name = set([protein_data[1] for protein_name,protein_data in dpo_test_dict.items()])

        WT_name_data_dict = {wt_name:{} for wt_name in WT_name}

        for protein_name,protein_data in dpo_test_dict.items():            
            WT_name_data_dict[protein_data[1]][protein_name]=protein_data

        return WT_name_data_dict
        
    def __len__(self):
        return len(self.WT_names)
    
    def get_data_dict(self):
        return self.WT_name_data_dict
    
    def __getitem__(self,idx):
        WT_name = self.WT_names[idx]
        protein_data_dict = self.WT_name_data_dict[WT_name]
        protein_data = protein_data_dict[list(protein_data_dict.keys())[0]]
        protein_structure = self.pdb_structure[self.pdb_idx_dict[protein_data[1]]]
        struc_seq = self.SaProt_foldseek_seq[self.pdb_idx_dict[protein_data[1]]]
        
        if self.is_AbDesign:
            Ab_seq = protein_data[2]
            Ag_seq = protein_data[6]
            full_seq = Ab_seq+Ag_seq
            assert len(struc_seq)==len(full_seq),"wrong "+protein_data[1]
            return {  
                'WT_name': protein_data[1], 
                'aa_seq_wt': full_seq, 
                'structure': protein_structure, 
                'struc_seq': struc_seq,
                'Ab_lenth':len(Ab_seq), 
                }
        else:
            return {  
                'WT_name': protein_data[1], 
                'aa_seq_wt': protein_data[2], 
                'structure': protein_structure, 
                'struc_seq': struc_seq,
                }



class ProteinDPOTrainDataset(Dataset):
    def __init__(self, dpo_train_dict,args):

        self.is_AbDesign = args.dataset=='AbDesign'
        max_lenth = max([len(data[0]+data[6])+10 for _,data in dpo_train_dict.items()]) if self.is_AbDesign else max([len(data[0])+10 for _,data in dpo_train_dict.items()])
        dataset = ProteinStructureDataset(args.pdb_path, self.is_AbDesign, max_lenth) 
        loader = DataLoader(dataset, batch_size=10000, shuffle=False)
        for batch in loader:
            pdb_structures = batch[0]
            pdb_filenames = batch[1]
            SaProt_foldseek_seq = batch[2]
            pdb_idx_dict = {pdb_filenames[i]: i for i in range(len(pdb_filenames))}
            break
        self.pdb_idx_dict = pdb_idx_dict
        self.pdb_structure = pdb_structures
        self.SaProt_foldseek_seq = SaProt_foldseek_seq
        
        self.dpo_train_dict = dpo_train_dict
        self.DPOpair_dict = self.process_train_pair_data(self.dpo_train_dict)
        self.DPOpair_name = list(self.DPOpair_dict.keys())

    def half_gap_pairs(self,input_dict):
        sorted_items = sorted(input_dict.items(), key=lambda x: x[1]) # Sort from small to large, take the front as disprefer
        gap_pairs = []
        half_disprefer,haf_prefer = int(len(sorted_items)/2),int(len(sorted_items)/2)
        if len(sorted_items):
            half_disprefer+=1
        for (protein_disprefer,_),(potein_prefer,_) in zip(sorted_items[:half_disprefer],sorted_items[haf_prefer:]):
            gap_pairs.append((protein_disprefer, potein_prefer))
        return gap_pairs

    def process_train_pair_data(self,dpo_train_dict):
        WT_name = set([protein_data[1] for protein_name,protein_data in dpo_train_dict.items()])
        WT_name_data_dict = {wt_name:{} for wt_name in WT_name}

        for protein_name,protein_data in dpo_train_dict.items():
            WT_name_data_dict[protein_data[1]][protein_name]=protein_data[3]

        all_gap_pairs = [] # [(protein_disprefer, potein_prefer)...]
        for WT_name,Mutant_DG_dict in WT_name_data_dict.items():
            all_gap_pairs += self.half_gap_pairs(Mutant_DG_dict)
        
        DPOpair_dict = {protein_disprefer+"___"+potein_prefer:(protein_disprefer, potein_prefer) for protein_disprefer, potein_prefer in all_gap_pairs}

        return DPOpair_dict

    def __len__(self):
        return len(self.DPOpair_name)
    
    def __getitem__(self,idx):
        pair_name = self.DPOpair_name[idx]
        protein_name_disprefer,protein_name_prefer = self.DPOpair_dict[pair_name]
        protein_data_prefer = self.dpo_train_dict[protein_name_prefer]
        protein_data_disprefer = self.dpo_train_dict[protein_name_disprefer]
        assert protein_data_prefer[1]==protein_data_disprefer[1], "wrong pair"
        protein_structure = self.pdb_structure[self.pdb_idx_dict[protein_data_prefer[1]]]
        struc_seq = self.SaProt_foldseek_seq[self.pdb_idx_dict[protein_data_prefer[1]]]
        
        
        if self.is_AbDesign:
            # concat[abseq,agseq]
            full_seq_prefer = protein_data_prefer[0]+protein_data_prefer[6]
            full_seq_disprefer = protein_data_disprefer[0]+protein_data_disprefer[6]
            full_seq_wt = protein_data_prefer[2]+protein_data_prefer[6]
                        
            assert len(struc_seq)==len(full_seq_wt) ,"wrong "+protein_data_prefer[1]
            assert len(full_seq_disprefer)==len(full_seq_prefer)
            
            return {
                'protein_name': idx,
                'aa_disprefer': full_seq_disprefer, 
                'aa_prefer':full_seq_prefer, 
                'WT_name': protein_data_prefer[1], 
                'aa_seq_wt': full_seq_wt, 
                'structure': protein_structure, 
                'struc_seq': struc_seq,
                "Ab_lenth": len(protein_data_prefer[2])
                }
        else:
            return {
                'protein_name': idx,
                'aa_disprefer': protein_data_disprefer[0], 
                'aa_prefer':protein_data_prefer[0], 
                'WT_name': protein_data_prefer[1], 
                'aa_seq_wt': protein_data_prefer[2], 
                'structure': protein_structure, 
                'struc_seq': struc_seq,
                }


        
        
class ProteinParetoDataset(Dataset):
    def __init__(self, dpo_iteration_dict,args):
        self.args = args
        self.dpo_iteration_dict = dpo_iteration_dict
        self.DPOpair_dict = self.process_train_pair_data(dpo_iteration_dict)
        self.DPOpair_name = list(self.DPOpair_dict.keys())

        max_lenth = max([len(data[0])+10 for _,data in dpo_iteration_dict.items()])
        dataset = ProteinStructureDataset(args.pdb_path, args.dataset=='AbDesign',max_lenth) 
        loader = DataLoader(dataset, batch_size=10000, shuffle=False)
        for batch in loader:
            pdb_structures = batch[0]
            pdb_filenames = batch[1]
            SaProt_foldseek_seq = batch[2]
            pdb_idx_dict = {pdb_filenames[i]: i for i in range(len(pdb_filenames))}
            break
        self.pdb_idx_dict = pdb_idx_dict
        self.pdb_structure = pdb_structures
        self.SaProt_foldseek_seq = SaProt_foldseek_seq

    
    def Normalization(self,scores):
        MAX=max(scores)
        MIN=min(scores)
        return (np.array(scores)-MIN)/(MAX-MIN)

    def get_projection_direction_group(self,points,Tolerance=False):
        points = np.asarray(points)
        if not Tolerance:
            assert np.all(points >= 0), "All point coordinates must be ≥ 0."
        def classify_sector(a, b):
            angles = np.arctan2(b, a)  
            degrees = np.rad2deg(angles)

            sectors = np.full_like(degrees, fill_value=2, dtype=int)  
            sectors[degrees < 30] = 1
            sectors[(degrees >= 60)] = 3
            return sectors
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        if self.args.dataset=='AAV&GFP':
            group_1 = classify_sector(y, x)
            group_2 = classify_sector(y, z)
        else:
            group_1 = classify_sector(x, y)
            group_2 = classify_sector(x, z)
        return [f"{a}{b}" for a, b in zip(group_1, group_2)]

    def half_gap_pairs(self,input_dict):
        # Sort from large to small, because the negative logarithm is used as the score, the smaller the score, the higher the probability.
        sorted_items = sorted(input_dict.items(), key=lambda x: x[1],reverse=True)  
        gap_pairs = []
        half_disprefer,haf_prefer = int(len(sorted_items)/2),int(len(sorted_items)/2)
        if len(sorted_items):
            half_disprefer+=1
        for (protein_disprefer,_),(potein_prefer,_) in zip(sorted_items[:half_disprefer],sorted_items[haf_prefer:]):
            gap_pairs.append((protein_disprefer, potein_prefer))
        return gap_pairs

    def compute_theta_over_90(self,input_scores, eps=1e-6):
        ref = torch.tensor([-1.0, -1.0, -1.0], device=input_scores.device).view(1, 3)
        ref = ref / ref.norm(p=2)  
        
        vecs = input_scores / (input_scores.norm(p=2, dim=1, keepdim=True) + eps)
        
        cos_theta = (vecs * ref).sum(dim=1).clamp(-1.0, 1.0)
        theta_rad = torch.acos(cos_theta)  
        theta_deg = torch.rad2deg(theta_rad)  
        theta_over_90 = theta_deg / 90.0  
        
        return theta_over_90.clamp(0.0, 1.0)

    def get_noise(self,pair_list,coords_dict):
        pair_direction_vector = torch.tensor(np.array([coords_dict[pre]-coords_dict[dispre] for dispre,pre in pair_list]))
        angle_over_90 = self.compute_theta_over_90(pair_direction_vector) # (Angle difference from x=y=z) / 90
        
        vote = [g.count("2") for g in self.get_projection_direction_group(-pair_direction_vector,Tolerance=True)] # Count several agents to reach consensus
        noise = (1-angle_over_90)*(((torch.tensor(vote)+0.3)/2).clamp(0.0, 1.0))
        return noise
    
    def process_train_pair_data(self,dpo_iteration_dict):
        
        WT_names = set([protein_data[1] for protein_name,protein_data in dpo_iteration_dict.items()])
        WTName_MultiAgentScores_dict={wt_name:{} for wt_name in WT_names}
        seqs=set()
        for prot_name,data_list in dpo_iteration_dict.items():
            wt_name = data_list[1]
            seq = data_list[0]
            if seq in seqs:
                continue
            seqs.add(seq)
            WTName_MultiAgentScores_dict[wt_name][prot_name]=(data_list[3],data_list[4],data_list[5])
            
        all_gap_pairs=[]
        noises=[]
        for wt_name,score_dict in WTName_MultiAgentScores_dict.items():
            # Normalize the three agent scores, calculate their coordinates relative to the origin, and group them according to their coordinates.
            prots = list(score_dict.keys())
            mpnn_scores = self.Normalization([score_dict[prot][0] for prot in prots])  
            esm_scores = self.Normalization([score_dict[prot][1] for prot in prots])  
            saprot_scores = self.Normalization([score_dict[prot][2] for prot in prots])  
            coords = list(zip(mpnn_scores,esm_scores,saprot_scores))
            distances = np.linalg.norm(np.array(coords), axis=1)  
            groups = self.get_projection_direction_group(coords)
            
            prot_coords_dict = {prot:np.array(coord) for prot,coord in zip(prots,coords)}
            prot_distance_dict = {prot:distance for prot,distance in zip(prots,distances)}
            
            # Group and construct pairs based on distance within the group, and calculate the noise of the pair based on the direction vector of the pair
            group_prot_dict = {g:[] for g in set(groups)}
            for index,group in enumerate(groups):
                group_prot_dict[group].append(prots[index])
            for group,prots in group_prot_dict.items():
                group_gap_pairs = self.half_gap_pairs({prot:prot_distance_dict[prot] for prot in prots})
                noise = self.get_noise(group_gap_pairs,prot_coords_dict)
                all_gap_pairs+=group_gap_pairs
                noises+=noise
        
        DPOpair_dict = {protein_disprefer+"___"+potein_prefer:(protein_disprefer, potein_prefer,noise) for (protein_disprefer, potein_prefer),noise in zip(all_gap_pairs,noises)}

        return DPOpair_dict

    def __len__(self):
        return len(self.DPOpair_name)
    
    def __getitem__(self,idx):
        pair_name = self.DPOpair_name[idx]
        protein_name_disprefer,protein_name_prefer,noise = self.DPOpair_dict[pair_name]
        protein_data_prefer = self.dpo_iteration_dict[protein_name_prefer]
        protein_data_disprefer = self.dpo_iteration_dict[protein_name_disprefer]
        assert protein_data_prefer[1]==protein_data_disprefer[1], "wrong pair"
        protein_structure = self.pdb_structure[self.pdb_idx_dict[protein_data_prefer[1]]]
        struc_seq = self.SaProt_foldseek_seq[self.pdb_idx_dict[protein_data_prefer[1]]]
        
        item = {
            'protein_name': idx,
            'aa_disprefer': protein_data_disprefer[0], 
            'aa_prefer':protein_data_prefer[0], 
            'WT_name': protein_data_prefer[1], 
            'aa_seq_wt': protein_data_prefer[2], 
            'structure': protein_structure, 
            'struc_seq': struc_seq,
            "noise":noise,
            }
        if self.args.dataset=='AbDesign':
            item['Ab_lenth']=protein_data_prefer[6]
            
        
        return item