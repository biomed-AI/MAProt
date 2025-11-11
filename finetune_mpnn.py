import os
import time
import json
import torch
import argparse
from tqdm.auto import tqdm
from types import SimpleNamespace
from common_utils.protein_oracle import ProteinMPNNOracle
from common_utils.protein_mpnn_utils import _scores

from dataset import *
from utils import *
from predictor import MultiAgentPredictor,MultiAgentDPOLoss, MultiAgentModelManager


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def train(device, policy_modelmanager, ref_modelmanager, log_file=None, save_dir=None, args=None,evaluate_target=None):

    # DPO Process
    loader_train,_,_ = get_dataloader(args,'train',os.path.join(args.data_path, 'dpo_train_dict_curated.pkl'))

    model_policy, optimizer = policy_modelmanager.get()
    loss_func = MultiAgentDPOLoss(args.beta)
    model_policy.to(device)
    model_policy.train()
    
    model_ref, _ = ref_modelmanager.get()
    model_ref.to(device)
    model_ref.eval()
    
    for epoch in range(args.epochs_dpo):
        print(f"\033[0;30;43m{time.strftime('%Y-%m-%d %H-%M-%S')} | [train] Epoch {epoch}\033[0m")
        # train:
        model_policy.train()
        bar = tqdm(loader_train)
        loss_sum = 0
        count = 0
        for batch in bar:
            loss = MultiAgent_process_batch(model_policy,model_ref, batch, loss_func,device, stage='train', optimizer=optimizer,is_AbDesign=args.dataset=='AbDesign')
            bar.set_description('loss: %.4f' % (loss.item()))
            count+=1
            loss_sum+=loss.item()
        loss_sum /=count
        log_msg = f"{time.strftime('%Y-%m-%d %H-%M-%S')} | [train] epoch {epoch} | Loss {loss_sum:.8f}"
        write_and_print(log_file,log_msg)
        
        model_policy.eval()
        
    torch.save(policy_modelmanager.state_dict(), os.path.join(save_dir, 'checkpoint', f'model_dpo.ckpt'))



def sample_sequence_for_pareto(device, model_policy, loader_test, sample_sequence_path, args):
    pdb_path = args.pdb_path    
    # sample
    config = {
        "sample seed":args.seed,
        "ouput_path":sample_sequence_path,
        "pdb_path":"",
        "num_seq_per_target":args.iteration_sample_num*2,
        "sampling_temp":str(args.sampling_temp),
        "backbone_noise":0.00,
        'sample_batchsize':10,
    }
    sample_num = args.iteration_sample_num
    WT_sample_sequence_list = {}
    count=0
    for batch in tqdm(loader_test): 
        assert len(batch['WT_name'])==1,"error test batch size!!!!!!!!!!"
        config['pdb_path'] = pdb_path+batch['WT_name'][0]
        ab_mask, each_agent_logits, _ = MultiAgent_process_batch(model_policy,None, batch, None, device,  stage='logit',is_AbDesign=args.dataset=='AbDesign')
        esm_logits = each_agent_logits[0][0]
        saprot_logits = each_agent_logits[1][0]
        
        prot_sample_sequence_path = os.path.join(sample_sequence_path,'seqs',batch['WT_name'][0][:-4]+".fa")     
        MultiAgent_sample_sequence(config,model_policy,esm_logits,saprot_logits,ab_mask)
              
        seq_dict = {}
        generated_seqs = set()
        for record_id,record_seq in read_fasta(prot_sample_sequence_path).items():
            if batch['WT_name'][0][:-4] in record_id:
                continue
            if record_seq not in generated_seqs:
                seq_dict[record_id]=str(record_seq).replace("/","")
            generated_seqs.add(record_seq)
            if len(generated_seqs)>=sample_num:
                break
            
        print("len generated ",batch['WT_name'][0]," seq:",len(seq_dict))
        WT_sample_sequence_list[batch['WT_name'][0]] = seq_dict
        
        
    return WT_sample_sequence_list


def Multi_Agent_Pareto_Distillation(device,policy_modelmanager,ref_modelmanager,
                                        args,log_file,save_dir,evaluate_target):
    _,loader_train_wt,train_wt_name_struc_seq_dict = get_dataloader(args,'train',os.path.join(args.data_path, 'dpo_train_dict_curated.pkl'))

    
    model_policy, optimizer = policy_modelmanager.get()
    loss_func = MultiAgentDPOLoss(args.beta)
    model_policy.to(device)
    model_policy.eval()
    
    model_ref, _ = ref_modelmanager.get()
    model_ref.to(device)
    model_ref.eval()
    
    
    # Get the -log probs of the wt sequence.
    policy_logprobs_dict = {}
    for batch in tqdm(loader_train_wt):
        each_agent_log_probs, each_agent_mask = MultiAgent_process_batch(model_policy,None, batch, None, device, stage='pareto')
        for i,wt_name in enumerate(batch['WT_name']):
            policy_logprobs_dict[wt_name]=[[log_probs[i,:,:],mask[i,:]] for log_probs,mask in zip(each_agent_log_probs, each_agent_mask)]
    
    # Sample sequence
    sample_sequence_path = save_dir+f"/Pareto_data/"
    os.makedirs(sample_sequence_path,exist_ok=True)
    train_WT_sample_sequence = sample_sequence_for_pareto(device,model_policy,loader_train_wt,sample_sequence_path,args)
      
    
    wtname_seq_dict={} # Get the sequence information of each wt for the following assembly generated data
    for batch in loader_train_wt:
        wtname_seq_dict[batch['WT_name'][0]]=batch['aa_seq_wt'][0]
    if args.dataset=='AbDesign':
        wtname_Ablenth_dict={} # Get the sequence information of each antigen for the following assembly generated data
        for batch in loader_train_wt:
            wtname_Ablenth_dict[batch['WT_name'][0]]=batch['Ab_lenth'][0]
            
    def get_score(log_probs_dict,agenti,S_sp,wt_name):
        log_probs,mask = log_probs_dict[wt_name][agenti]
        log_probs = log_probs.repeat(len(S_sp), 1)
        mask = mask.repeat(len(S_sp), 1)
        return _scores(S_sp,log_probs,mask)
    
    
    generated_data={}
    for wt_name,gen_seq_dict in train_WT_sample_sequence.items():
        struc_seq = train_wt_name_struc_seq_dict[wt_name]
        mpnn_mask,esm_mask,saprot_mask=[agent_res[1] for agent_res in policy_logprobs_dict[wt_name]]
        S_sp = [[],[],[]] # mpnn, esm, saprot
        gen_seqids = list(gen_seq_dict.keys())
        for gen_seqid in gen_seqids:
            gen_seq = gen_seq_dict[gen_seqid]
            S_sp[0].append(mpnn_seq_to_tensor(str(gen_seq),mpnn_mask,device).unsqueeze(0))
            S_sp[1].append(esm_seq_to_tensor(str(gen_seq),esm_mask,model_policy.esm_alphabet,device).unsqueeze(0))
            S_sp[2].append(saprot_seq_to_tensor(str(gen_seq),struc_seq,saprot_mask,model_policy.saprot_tokenizer,device).unsqueeze(0))
        S_sp = [torch.concat(S_sp_i,dim=0) for S_sp_i in S_sp]
        
        mpnn_policy_scores = get_score(policy_logprobs_dict,0,S_sp[0],wt_name).detach().cpu().numpy().tolist()
        esm_policy_scores = get_score(policy_logprobs_dict,1,S_sp[1],wt_name).detach().cpu().numpy().tolist()
        saprot_policy_scores = get_score(policy_logprobs_dict,2,S_sp[2],wt_name).detach().cpu().numpy().tolist()
        
        mpnn_score_dict = {gen_seqid:p_score for gen_seqid,p_score in zip(gen_seqids,mpnn_policy_scores)}
        esm_score_dict = {gen_seqid:p_score for gen_seqid,p_score in zip(gen_seqids,esm_policy_scores)}
        saprot_score_dict = {gen_seqid:p_score for gen_seqid,p_score in zip(gen_seqids,saprot_policy_scores)}
        
        for gen_seqid,gen_seq in gen_seq_dict.items():
            data=[gen_seq,wt_name,wtname_seq_dict[wt_name],mpnn_score_dict[gen_seqid],
                    esm_score_dict[gen_seqid],saprot_score_dict[gen_seqid]]
            if args.dataset=='AbDesign': 
                data.append(wtname_Ablenth_dict[wt_name])
            generated_data[wt_name+gen_seqid]=data
        

    train_iteration_dataset = ProteinParetoDataset(generated_data,args)
    loader_train_iteration = DataLoader(train_iteration_dataset, batch_size=args.batch_size, shuffle=True)
               
    pickle.dump(train_iteration_dataset.DPOpair_dict,open(os.path.join(sample_sequence_path, 'train_iteration_dpo_pair.pkl'),'wb'))
        
    for epoch in range(args.iteration_epoch):
        print(f"\033[0;30;43m{time.strftime('%Y-%m-%d %H-%M-%S')} | [train] Iteration Epoch {epoch}\033[0m")

        model_policy.train()
        bar = tqdm(loader_train_iteration)
        loss_sum = 0
        count = 0
        for batch in bar:
            loss = MultiAgent_process_batch(model_policy,model_ref, batch, loss_func,device, stage='iteration_train', optimizer=optimizer,is_AbDesign=args.dataset=='AbDesign')
            bar.set_description('loss: %.4f' % (loss.item()))
            count+=1
            loss_sum+=loss.item()
        loss_sum /=count
        log_msg = f"{time.strftime('%Y-%m-%d %H-%M-%S')} | [train] Iteration epoch {epoch} | Loss {loss_sum:.8f}"
        write_and_print(log_file,log_msg)
        
        torch.cuda.empty_cache()
        
    torch.save(policy_modelmanager.state_dict(), os.path.join(save_dir, 'checkpoint', f'model.ckpt'))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or infer with MAProt model.")
    parser.add_argument('--config', type=str, default="./config/", help="Path to configuration JSON file.")
    parser.add_argument('--output_path', type=str, default="./results/")
    args = parser.parse_args()
    param_s = args.__dict__
    param = json.loads(open(args.config, 'r').read())
    merged_params = {**param, **param_s}
    args = argparse.Namespace(**merged_params)
    set_seed(args.seed)

    # Setup directories and logging
    job_name = os.path.splitext(os.path.basename(args.config))[0]
    save_dir = os.path.join(args.output_path, job_name)
    check_dir(os.path.join(save_dir, 'checkpoint'))
    log_file = open(os.path.join(save_dir, "train_log.txt"), 'a+',buffering=1)
    with open(os.path.join(save_dir, 'train_config.json'), 'w') as fout:
        json.dump(args.__dict__, fout, indent=2)



    ref_modelmanager = MultiAgentModelManager(config=args, model_factory=MultiAgentPredictor).to('cpu')
    ref_modelmanager.load_mpnn_state_dict(torch.load(args.ckpt_path, map_location='cpu'))


    print('Stage1 Single-Agent Preference Alignment')
    policy_modelmanager = MultiAgentModelManager(config=args, model_factory=MultiAgentPredictor).to('cpu')
    policy_modelmanager.load_mpnn_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
    
    train(device=device, policy_modelmanager=policy_modelmanager,ref_modelmanager=ref_modelmanager, 
          log_file=log_file, save_dir=save_dir, args=args,
          evaluate_target=args.dataset)
    
        
    print('Stage2 Pareto-based Multi-Agent Preference Agreement')
    policy_modelmanager = MultiAgentModelManager(config=args, model_factory=MultiAgentPredictor).to('cpu')
    policy_modelmanager.load_state_dict_inference(torch.load(os.path.join(save_dir, 'checkpoint', f'model_dpo.ckpt'), map_location='cpu'))
    
    Multi_Agent_Pareto_Distillation(device=device, policy_modelmanager=policy_modelmanager,ref_modelmanager=ref_modelmanager, 
            log_file=log_file, save_dir=save_dir, args=args,
            evaluate_target=args.dataset,)
