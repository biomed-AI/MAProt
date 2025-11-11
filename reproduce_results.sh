python finetune_mpnn.py --config ./config/megascale.json
python evaluate.py --config ./config/megascale.json

python finetune_mpnn.py --config ./config/AffinityDesign.json
python evaluate.py --config ./config/AffinityDesign.json


python finetune_mpnn.py --config ./config/GFP_hard.json
python evaluate.py --config ./config/GFP_hard.json

python finetune_mpnn.py --config ./config/GFP_medium.json
python evaluate.py --config ./config/GFP_medium.json