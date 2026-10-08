from detrex.config import get_config
from detectron2.config import LazyCall as L
from .models.hqp_detr_r50 import model
dataloader = get_config("common/data/flux_proposal.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/ws_pascalvoc_schedule.py").flux_scheduler_12ep
train = get_config("common/train.py").train
train.init_checkpoint = "./models/R-50.pkl"
train.output_dir = "./output/"
train.max_iter = 3000
train.eval_period = 3000
train.log_period = 20
train.checkpointer.period = 3000
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2
train.device = "cuda"
model.device = train.device
optimizer.lr = 1.25e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1
dataloader.train.num_workers = 32
dataloader.train.total_batch_size = 40
dataloader.evaluator.output_dir = "./output/cascadeHQP-DETR"