import os
import time
import json
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import copy

from dataset import *
from utils import *
from predictor import MultiAgentPredictor, MultiAgentModelManager
from types import SimpleNamespace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_design(device, policy_modelmanager ,save_dir, args, 
                evaluate_target=None,
                reward_model_eval=None,the_folding_model=None,evalRunner=None):
    pdb_path = args.pdb_path    
    loader_test = get_dataloader(args,'test',os.path.join(args.data_path, 'dpo_test_dict.pkl'))

    model_policy, _ = policy_modelmanager.get()
    model_policy.to(device)
    model_policy.eval()
    
    
    sample_sequence_path = save_dir+f"/sample_results/"
    sample_num=args.sample_num
    config = {
        "sample seed":args.seed,
        "ouput_path":sample_sequence_path,
        "pdb_path":"",
        "num_seq_per_target":sample_num,
        "sampling_temp":str(args.sampling_temp), # 0.1
        "sample_batchsize":1,
    }
    for batch in tqdm(loader_test):
        assert len(batch['WT_name'])==1,"error test batch size!!!!!!!!!!"
        config['pdb_path'] = pdb_path+batch['WT_name'][0]
        ab_mask, each_agent_logits, each_agent_mask = MultiAgent_process_batch(model_policy,None, batch, None, device, stage='logit',is_AbDesign=args.dataset=='AbDesign')
        esm_logits = each_agent_logits[0][0]
        saprot_logits = each_agent_logits[1][0]
        
    
        prot_sample_sequence_path = os.path.join(sample_sequence_path,'seqs',batch['WT_name'][0][:-4]+".fa")
        if os.path.exists(prot_sample_sequence_path):
            continue
        
        if evaluate_target=='AAV&GFP':
            for i in range(7):
                MultiAgent_sample_sequence(config,model_policy,esm_logits,saprot_logits,ab_mask)
                generated_seqs = []
                for record_id,record_seq in read_fasta(prot_sample_sequence_path).items():
                    if batch['WT_name'][0][:-4] in record_id:
                        continue
                    generated_seqs.append(record_seq)
                if len(set(generated_seqs))>=sample_num:
                    break
                else:
                    config['num_seq_per_target']=config['num_seq_per_target']*2
        else:
            MultiAgent_sample_sequence(config,model_policy,esm_logits,saprot_logits,ab_mask)
        
    
    with open(os.path.join(sample_sequence_path, 'evaluate_config.json'), 'w') as fout:
        json.dump(args.__dict__, fout, indent=2)
    torch.cuda.empty_cache()
    
    if evaluate_target=='megascale':
        repeat_num=sample_num
        results_merge = []
        
        for _, batch in tqdm(enumerate(loader_test)):
            
            print(batch['WT_name'][0][:-4])
            X, S, mask, chain_M, residue_idx, chain_encoding_all = featurize_test_mpnn(batch,device)
            X = X.repeat(repeat_num, 1, 1, 1)
            mask = mask.repeat(repeat_num, 1)
            chain_M = chain_M.repeat(repeat_num, 1)
            residue_idx = residue_idx.repeat(repeat_num, 1)
            chain_encoding_all = chain_encoding_all.repeat(repeat_num, 1)
            prot_sample_sequence_path = os.path.join(sample_sequence_path,'seqs',batch['WT_name'][0][:-4]+".fa")
            
            S_sp = []
            for record_id,record_seq in read_fasta(prot_sample_sequence_path).items():
                if batch['WT_name'][0][:-4] in record_id:
                    continue
                S_sp.append(mpnn_seq_to_tensor(str(record_seq),torch.ones(len(record_seq)),device).unsqueeze(0))
            S_sp = torch.concat(S_sp,dim=0)

            mask_for_loss = mask*chain_M
            results_list = cal_rmsd(S_sp, S, batch, the_folding_model, pdb_path, mask_for_loss, sample_sequence_path)
            torch.cuda.empty_cache()
            
            # Calculate the Drakes model prediction dG
            with torch.no_grad():
                ddg_pred_eval = reward_model_eval(X, S_sp, mask, chain_M, residue_idx, chain_encoding_all)
                ddg_pred_eval = ddg_pred_eval.detach().cpu().numpy()
            
            results_list = pd.concat(results_list)
            results_list['rewards_eval']=ddg_pred_eval
            results_merge.append(results_list)
            
            torch.cuda.empty_cache()
        
        # process results
        results_merge = pd.concat(results_merge)
        rewards_eval = np.array(results_merge['rewards_eval'].tolist())
        print(rewards_eval)
        print('Mean reward: ', np.mean(rewards_eval), "Positive reward prop %f"%np.mean(rewards_eval>0), 'median reward: ', np.median(rewards_eval))
        avg_rmsd = results_merge['gen_true_bb_rmsd'].mean()
        mid_rmsd = results_merge['gen_true_bb_rmsd'].median()
        rmsd_rate = results_merge['gen_true_bb_rmsd'].apply(lambda x: 1 if x < 2 else 0).mean()
        print('Median gen_true RMSD: ', mid_rmsd, 'Mean gen_true RMSD: ', avg_rmsd, 'Good RMSD prop: ', rmsd_rate)
        results_merge['success'] = (results_merge['gen_true_bb_rmsd'] < 2) & (results_merge['rewards_eval'] > 0)
        success_rate = results_merge['success'].mean()
        print('success rate: ', success_rate)
        
        results_merge.to_csv(sample_sequence_path+'results_merge.csv')
        results_dict = {
                        'Mean Reward': np.mean(rewards_eval), 
                        'Positive Reward Proportion': np.mean(rewards_eval>0), 
                        'Median Reward': np.median(rewards_eval), 
                        'Median gen_true RMSD': mid_rmsd, 
                        'Mean gen_true RMSD': avg_rmsd,
                        'Good RMSD Proportion': rmsd_rate,
                        'Success Rate': success_rate} 
        results_summary = pd.DataFrame.from_dict(results_dict, orient='index', columns=['Value'])
        results_summary.to_csv(save_dir+'results_summary.csv')

    elif evaluate_target=='AAV&GFP':
        
        for _, batch in enumerate(loader_test):
            prot_sample_sequence_path = os.path.join(sample_sequence_path,'seqs',batch['WT_name'][0][:-4]+".fa")
            
            generated_seqs = []
            for record_id,record_seq in read_fasta(prot_sample_sequence_path).items():
                if batch['WT_name'][0][:-4] in record_id:
                    continue
                generated_seqs.append(record_seq)
        results_merge,results_dict = evalRunner.evaluate_sequences(generated_seqs)
        print(results_dict)
        results_merge.to_csv(sample_sequence_path+'results_merge.csv')
        results_summary = pd.DataFrame.from_dict(results_dict, orient='index', columns=['Value'])
        results_summary.to_csv(save_dir+f'results_summary.csv')
            
    elif evaluate_target == "AbDesign":
        result_merge={"Ab_seq":[],"Ag_seq":[],"wt_name":[],"delta_affinity":[],"Ab_recovery":[]}
        Ab_diversity,Ab_aar=[],[]
        for _, batch in tqdm(enumerate(loader_test)):
            prot_sample_sequence_path = os.path.join(sample_sequence_path,'seqs',batch['WT_name'][0][:-4]+".fa")
            
            ab_lenth = batch['Ab_lenth'][0]
            wt_seq = batch['aa_seq_wt'][0]
            wt_seq = [(wt_seq[:ab_lenth],wt_seq[ab_lenth:])]
            
            generated_seqs = []
            for record_id,record_seq in read_fasta(prot_sample_sequence_path).items():
                if batch['WT_name'][0][:-4] in record_id:
                    continue
                generated_seq = str(record_seq).replace("/","")
                generated_seqs.append((generated_seq[:ab_lenth],generated_seq[ab_lenth:]))
            res_dict = evalRunner.evaluate_sequences(generated_seqs,wt_seq)
            
            result_merge['Ab_seq']+=[ab_seq for ab_seq,ag_seq in generated_seqs]
            result_merge['Ag_seq']+=[ag_seq for ab_seq,ag_seq in generated_seqs]
            result_merge['wt_name']+=[batch['WT_name'][0] for _ in range(len(generated_seqs))]
            
            result_merge['delta_affinity']+=res_dict['delta_affinity']
            result_merge['Ab_recovery']+=res_dict['recovery']
            
            Ab_diversity+=res_dict['diversity']
            Ab_aar+=res_dict['aar']
            
        result_merge=pd.DataFrame(result_merge)
        result_merge['success'] = (result_merge['delta_affinity'] < 0) & (result_merge['Ab_recovery'] > 0.7)
        result_merge.to_csv(sample_sequence_path+'results_merge.csv')
        
        
        delta_affinity = np.array(result_merge['delta_affinity'].tolist())
        Ab_recovery = np.array(result_merge['Ab_recovery'].tolist())
        
        success_rate = result_merge['success'].mean()
        results_dict = {
                        'Mean delta affinity': np.mean(delta_affinity), 
                        'Median delta affinity': np.median(delta_affinity), 
                        'Positive Reward Proportion': np.mean(delta_affinity<0), 
                        
                        'Median seq recovery': np.median(Ab_recovery), 
                        'Mean seq recovery': np.mean(Ab_recovery),
                        "seq recovery>70% proportion":np.mean(Ab_recovery>=0.7),
                        
                        'Mean Ab diversity': np.mean(Ab_diversity), 
                        'Median Ab diversity': np.median(Ab_diversity), 
                        
                        'Mean Ab aar': np.mean(Ab_aar), 
                        'Median Ab aar': np.median(Ab_aar), 
                        
                        'Success Rate': success_rate} 
        print(results_dict)
        results_summary = pd.DataFrame.from_dict(results_dict, orient='index', columns=['Value'])
        results_summary.to_csv(save_dir+'results_summary.csv')
                
    else:
        print("no specific evaluate")
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or infer with MAProt model.")
    parser.add_argument('--config', type=str, default="./config/AbDesign.json", help="Path to configuration JSON file.")
    parser.add_argument('--output_path', type=str, default="./results/")
    args = parser.parse_args()
    
    
    param_s = args.__dict__
    param = json.loads(open(args.config, 'r').read())
    merged_params = {**param, **param_s}
    args = argparse.Namespace(**merged_params)
    
    set_seed(args.seed)

    job_name = os.path.splitext(os.path.basename(args.config))[0]
    print(f"Starting Evaluate [{job_name}] | task name [{args.dataset}]")
    save_dir = os.path.join(args.output_path, job_name)

    policy_modelmanager = MultiAgentModelManager(config=args, model_factory=MultiAgentPredictor).to('cpu')
    policy_modelmanager.load_state_dict_inference(torch.load(os.path.join(save_dir, 'checkpoint', f'model.ckpt'), map_location='cpu'))

    if args.dataset == 'megascale':
        from common_utils import folding_model
        from common_utils.protein_oracle import ProteinMPNNOracle
        
        # To calculate the model predicted dG, from drakes
        reward_model_eval = ProteinMPNNOracle(node_features=128,
                            edge_features=128,
                            hidden_dim=128,
                            num_encoder_layers=3,
                            num_decoder_layers=3,
                            k_neighbors=30,
                            dropout=0.1,
                            )
        reward_model_eval.to(device)
        reward_model_eval.load_state_dict(torch.load(args.reward_model_path)['model_state_dict'])
        reward_model_eval.finetune_init()
        reward_model_eval.eval()
        
        
        # To calculate scRMSD, initialize emsfold
        folding_cfg = {
            'seq_per_sample': 1,
            'folding_model': 'esmf',
            'own_device': False,
            'pt_hub_dir': os.path.join("../tmp", '.cache/torch/'),
            'colabfold_path': os.path.join("../tmp", 'colabfold-conda/bin/colabfold_batch') # for AF2
        }
        folding_cfg = SimpleNamespace(**folding_cfg)
        the_folding_model = folding_model.FoldingModel(folding_cfg)
        
        test_design(device=device, policy_modelmanager=policy_modelmanager,save_dir=save_dir, args=args,
                    evaluate_target='megascale',
                    the_folding_model=the_folding_model,reward_model_eval=reward_model_eval)
    
    
    elif args.dataset == 'AAV&GFP':
        from common_utils.GFP_evaluate_utils import EvalRunner
        config = {
            "gt_csv_path":args.groundtruth_data_path,
            "oracle_dir":args.GGS_oracle_path,
            'train_data_path':os.path.join(args.data_path, 'dpo_train_dict_curated.pkl'),
            'batch_size':128,
        }
        evalRunner = EvalRunner(config,device)
        test_design(device=device, policy_modelmanager=policy_modelmanager,save_dir=save_dir, args=args,
                    evaluate_target='AAV&GFP',
                    evalRunner=evalRunner
                    )
    elif args.dataset == "AbDesign":
        from common_utils.AbDesign_evaluate_utils import AbDesignEvalRunner
        
        evalRunner = AbDesignEvalRunner(args.reward_model_path,
                                        device)
        test_design(device=device, policy_modelmanager=policy_modelmanager,save_dir=save_dir, args=args,
                    evaluate_target='AbDesign',
                    evalRunner=evalRunner
                    )