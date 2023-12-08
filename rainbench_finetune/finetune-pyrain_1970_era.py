#!/usr/bin/env python
# coding: utf-8

# In[1]:



# # Finetuning Pyrain
# 
# This notebook will demonstrate how to finetune climaX on the RainBench preciptation data and code provided by [Pyrain](https://github.com/FrontierDevelopmentLab/PyRain/tree/master). This notebook is based on the [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) framework and can be adapted to other datasets.

# ## Data
# The precipitation data provided by Pyrain can be downloaded [here](https://console.cloud.google.com/storage/browser/aaai_release).
# 
# ### Input
# For this notebook, we will use RainBench data from both Simsat and ERA5. Refer to the [Rainbench](https://arxiv.org/abs/2012.09670) paper for the complete list of variables. The input is composed of time series over a 12 hour period, sampled every 6 hours by default.
# 
# ![input.png](images/input.png)
# 
# The shape is $T \times V \times H \times W$, where $T$ is the number of input time steps, $V$ is the number of variables, and $H$, $W$ the spatial resolution (32 x 64 for 5.625°).
# 
# ### Output
# The network will be trained to predict the precipitation at several lead times.

# # Setup
# 
# Create a conda environment for training ClimaX. Installation guide can be found [here](https://microsoft.github.io/ClimaX/install/). This notebook also requires installing additional packages:
# 
# ```bash
# pip install dill ## for loading pyrain data
# pip install deepspeed ## for efficient training
# ```

# # Configs
# 
# We can use a hparams dictionary to store all the hyperparameters for initializing the dataloaders and training. Remeber to change the paths for `data_paths` and `log_path`:

# In[2]:


hparams = {
    'seed': 2020,
    'sources': 'era',  # options: 'simsat_era', 'simsat', 'era', 'era16_3'
    'imerg': True,  # options: True (predict IMERG), False (predict ERA5)
    'grid': 5.625, 
    'time_history': 3,
    'sample_time_window': 12,
    'sample_freq': 6,
    'forecast_time_window': 8766,
    'forecast_freq': 4383,
    'inc_time': True,
    'data_paths': [  # where precipitation data is stored
        '/home/allen/data/rainbench/era5625_aaai/era5625_us.dill', 
        '/home/allen/data/rainbench/imerg5625/storm_data_us.dill', 
        '/home/allen/data/rainbench/simsat5625/simsat5625.dill',
        '/home/allen/data/rainbench/nino5625/nino5625.dill'
    ],
    'norm_path': 'pyrain/normalize.json',
    'log_path': '/home/allen/ckpts/',  # where the checkpoints should go
    'gpus': 1,
    'use_amp': True,
    'batch_size': 32,
    'lr': 5e-05,
    'num_workers': 4,
    # 'strategy': 'deepspeed_stage_2', # Deepspeed not available in interactive environments
    'strategy': None,
    'acc_grad': 1,
    'version': 'pyrain-finetune-template',
    'plot': False,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'weight_decay': 1e-05,
    'warmup_epochs': 5,
    'max_epochs': 20,
    'warmup_start_lr': 1e-08,
    'eta_min': 0.00000001
}


# # Datamodule
# 
# We will first define a datamodule that will load the data and prepare it for training.
# 

# In[3]:


from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from pyrain.dataset import RainbenchDataset
from pyrain.collect_data import write_data_config, read_normalization_stats
from pyrain.utils import get_local_shift, collate_fn


# In[4]:


class RainbenchDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.data_dir, self.partition_conf, self.sample_conf = write_data_config(hparams)
        self.normalizer = read_normalization_stats(hparams['norm_path'])

        self.train_dataset = RainbenchDataset(
            datapath=self.data_dir,
            partition_conf=self.partition_conf,
            partition_type="range",
            partition_selected="train",
            sample_conf=self.sample_conf)

        self.val_dataset = RainbenchDataset(
            datapath=self.data_dir,
            partition_conf=self.partition_conf,
            partition_type="range",
            partition_selected="valid",
            sample_conf=self.sample_conf)

        self.test_dataset = RainbenchDataset(
            datapath=self.data_dir,
            partition_conf=self.partition_conf,
            partition_type="range",
            partition_selected="test",
            sample_conf=self.sample_conf)

        time_shift = None
        if hparams['inc_time']:
            time_shift = get_local_shift(hparams['grid'], self.train_dataset.dataset)
        self.collate = lambda x: collate_fn(x, hparams, self.normalizer, time_shift)

    

    def get_lat2d(self, grid):
        if grid == 5.625:
            lat2d = self.val_dataset.dataset['era5625/lat2d']
        else:
            lat = np.linspace(-89.296875, 89.296875, 128)
            lat2d = np.expand_dims(lat, axis=1).repeat(256, 1)
        return lat2d
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=hparams['batch_size'], num_workers=hparams['num_workers'], collate_fn=self.collate, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=hparams['batch_size'], num_workers=hparams['num_workers'], collate_fn=self.collate, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=hparams['batch_size'], num_workers=hparams['num_workers'], collate_fn=self.collate, shuffle=False)


# ## Initialize datamodule

# In[5]:


# define DataModule
datamodule = RainbenchDataModule()
lat2d = datamodule.get_lat2d(hparams['grid'])


# # Model architecture
# 
# We need to modify the original ClimaX architecture for this task because:
# - We want to predict only precipitation, while the original architecture predicts all input variables.
# - We must aggregate multiple time steps into a single input and add a time embedding to the input. This was not taken into account by the original model
# 
# We base this new architecture on the original architecture.
# 
# <!-- Since the [input](#Input) and [output](#Output) are different from the original climaX task, we will need to modify the model architecture. We will use the same encoder and decoder (no freezing), but change the head to predict just one variable (precipitation) at a time. We also must aggregate multiple time steps into a single input and add a time embedding to the input. -->

# In[6]:


from climax.arch import ClimaX
import torch
import torch.nn as nn
import numpy as np
from climax.utils.pos_embed import get_1d_sincos_pos_embed_from_grid


# In[7]:


class ClimaXRainBench(ClimaX):
    def __init__(
        self,
        default_vars,
        out_vars,
        img_size=[5, 12],
        time_history=1,
        patch_size=1,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
        parallel_patch_embed=False,
        freeze_encoder=False,
    ):
        assert out_vars is not None

        super().__init__(
            default_vars,
            img_size,
            patch_size,
            embed_dim,
            depth,
            decoder_depth,
            num_heads,
            mlp_ratio,
            drop_path,
            drop_rate,
            parallel_patch_embed
        )

        self.out_vars = out_vars
        self.time_history = time_history
        self.freeze_encoder = freeze_encoder

        # used to aggregate multiple timesteps in the input
        self.time_pos_embed = nn.Parameter(torch.zeros(1, time_history, embed_dim), requires_grad=True)
        self.time_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.time_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)

        # initialize time embedding
        time_pos_embed = get_1d_sincos_pos_embed_from_grid(self.time_pos_embed.shape[-1], np.arange(self.time_history))
        self.time_pos_embed.data.copy_(torch.from_numpy(time_pos_embed).float().unsqueeze(0))

        # overwrite ClimaX
        # replace head for this task, since output is different
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, patch_size**2))
        self.head = nn.Sequential(*self.head)

        if freeze_encoder:
            for name, p in self.blocks.named_parameters():
                name = name.lower()
                # we do not freeze the norm layers, as suggested by https://arxiv.org/abs/2103.05247
                if 'norm' in name:
                    continue
                else:
                    p.requires_grad_(False)


    def forward_encoder(self, x: torch.Tensor, lead_times: torch.Tensor, variables):
        # x: `[B, T, V, H, W]` shape
        if isinstance(variables, list):
            variables = tuple(variables)
        
        b, t, _, _, _ = x.shape
        x = x.flatten(0, 1)  # BxT, V, H, W
        
        # tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)

        if self.parallel_patch_embed:
            x = self.token_embeds(x, var_ids)  # BxT, V, L, D
        else:
            for i in range(len(var_ids)):
                id = var_ids[i]
                embeds.append(self.token_embeds[id](x[:, i : i + 1]))
            x = torch.stack(embeds, dim=1)  # BxT, V, L, D

        # add variable embedding
        var_embed = self.get_var_emb(self.var_embed, variables)
        x = x + var_embed.unsqueeze(2)  # BxT, V, L, D

        # variable aggregation
        x = self.aggregate_variables(x)  # BxT, L, D

        # add pos embedding
        x = x + self.pos_embed

        # add time embedding
        # time emb: 1, T, D
        x = x.unflatten(0, sizes=(b, t)) # B, T, L, D
        x = x + self.time_pos_embed.unsqueeze(2)

        # add lead time embedding
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1)) # B, D
        lead_time_emb = lead_time_emb.unsqueeze(1).unsqueeze(2)
        x = x + lead_time_emb # B, T, L, D

        x = x.flatten(0, 1)  # BxT, L, D

        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x) # BxT, L, D  
        x = x.unflatten(0, sizes=(b, t)) # B, T, L, D

        time_query = self.time_query.repeat_interleave(x.shape[0], dim=0)
        # run time_agg for each L, so that the final output is B, L, D
        agg_x = torch.empty(0, dtype=x.dtype).to(x.device)
        for i in range(x.shape[2]):
            agg_x_i, _ = self.time_agg(time_query, x[:, :, i, :], x[:, :, i, :])
            agg_x = torch.cat((agg_x, agg_x_i), dim=1)

        return agg_x    # B, L, D


    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, patch_size**2)
        return imgs: (B, 1, H, W)
        """
        p = self.patch_size
        c = 1
        h = self.img_size[0] // p if h is None else h // p
        w = self.img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs
    

    def forward(self, x, y, lead_times, variables, out_variables, metric, lat):
        out_transformers = self.forward_encoder(x, lead_times, variables)  # B, L, D
        preds = self.head(out_transformers)  # B, L, p*p
        preds = self.unpatchify(preds) # B, 1, H, W

        if metric is None:
            loss = None
        else:
            loss = [m(preds, y, out_variables, lat) for m in metric]

        return loss, preds


# ## Initialize the model

# In[8]:


climax_var_order = ["lsm", "orography", "lat2d", "t2m", "z-500", "z-850", "t-500", "t-850", "q-500", "q-850"]
categories = hparams['categories']
sorted_vars = sorted(categories['input'], key=lambda x: climax_var_order.index(x) if x in climax_var_order else len(climax_var_order))
sorted_vars.append('lead_time')
net = ClimaXRainBench(
    default_vars=sorted_vars,
    out_vars=categories['output'],
    time_history=3, # the number of input timesteps
)


# ## Load pretrained weights
# 
# We provide several pretrained models that can be used for finetuning on a different task. For more details, please check out the load_climax function.
# 
# In this particular benchmark, we will load ClimaX-v1 pretrained on CMIP6 at 5.625 deg. Note that we also pass our custom network to this function. This may not be needed for tasks that can use the original ClimaX architecture for finetuning.

# In[9]:


from climax.loaders import load_climax

net = load_climax(
    version=1,
    pretraining_data='cmip6',
    resolution=hparams['grid'],
    preset_net=net
)


# The output tells us the following information:
# - The last layer of the head was replaced, so its keys were removed from the checkpoint, and new keys were added, hence the IncompatibleKeys.
# - The SIMSAT + ERA5 dataset only has 26 variables, and the lead_time makes it 27. Therefore, the token_embeds 28 to 47 were removed.
# - The input variables are different, so the var_embed was replaced.
# - time_pos_embed, time_query, time_agg were not part of the original ClimaX model, but they were added to take multiple timesteps as input.

# # Forecast Module
# 
# We will now define the forecast module that will be used for training. This module will be initialized with the pretrained climaX model and will be finetuned on the precipitation data.

# In[10]:


import json
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer, loggers
from climax.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from climax.utils.metrics import (
    mse,
    lat_weighted_mse,
    lat_weighted_nrmse, 
    lat_weighted_rmse,
)
from climax.utils.pos_embed import interpolate_pos_embed
from typing import Any
from pyrain.metrics import eval_loss, define_loss_fn, collect_outputs
from deepspeed.ops import adam


# In[11]:


class RainForecastModule(LightningModule):
    def __init__(
        self,
        net,
        lat2d,
        normalizer,
        pretrained_path: str = "",
    ):
        super().__init__()
        self.categories = hparams['categories']
        self.net = net
        if len(pretrained_path) > 0:
            self.load_pretrained_weights(pretrained_path)

        self.lead_times = hparams['lead_times']
        self.lat, self.lon = hparams['latlon']
        self.test_step_outputs = []
        self.val_step_outputs = []
        self.version = hparams["version"]
        self.normalizer = normalizer
        
        self.weights_lat, self.loss = define_loss_fn(lat2d)
        self.lat = lat2d[0][:,0]

    def load_pretrained_weights(self, pretrained_path):
        if pretrained_path.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_path, map_location=torch.device("cpu"))
        else:
            checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))

        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint["state_dict"]
        # interpolate positional embedding
        interpolate_pos_embed(self.net, checkpoint_model, new_size=self.net.img_size)

        state_dict = self.state_dict()
        if self.net.parallel_patch_embed:
            if "token_embeds.proj_weights" not in checkpoint_model.keys():
                raise ValueError(
                    "Pretrained checkpoint does not have token_embeds.proj_weights for parallel processing. Please convert the checkpoints first or disable parallel patch_embed tokenization."
                )

        for k in list(checkpoint_model.keys()):
            if "channel" in k:
                checkpoint_model[k.replace("channel", "var")] = checkpoint_model[k]
                del checkpoint_model[k]

            if "head" in k:
                print(f"Removing key {k} from pretrained checkpoint.")
                del checkpoint_model[k]

        for k in list(checkpoint_model.keys()):
            if k not in state_dict.keys() or checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)


    def training_step(self, batch: Any, batch_idx: int):
        x, y, lead_times = batch

        loss_dict, p = self.net.forward(x, y, lead_times, self.categories['input'], self.categories['output'], [lat_weighted_mse], lat=self.lat)

        loss_dict = loss_dict[0]
        for var in loss_dict.keys():
            self.log(
                "train/" + var,
                loss_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        loss = loss_dict['loss']

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, lead_times = batch

        _, pred = self.net.forward(
            x,
            y,
            lead_times,
            self.categories['input'],
            self.categories['output'],
            metric=None,
            lat=self.lat,
        )

        results = eval_loss(pred, y, lead_times, self.loss, self.lead_times, phase='val', target_v=self.categories['output'][0], normalizer=self.normalizer)

        self.val_step_outputs.append(results)
        return results
    
    def on_validation_epoch_end(self):
        node_loss = collect_outputs(self.val_step_outputs, False)
        self.val_step_outputs.clear()  # free memory

        if isinstance(node_loss, list):
            node_loss = node_loss[0]
    
        all_losses = self.all_gather(node_loss)
        mean_losses = {k: float(torch.mean(x)) for k, x in all_losses.items()}

        # log mean losses
        for var in mean_losses.keys():
            self.log(
                "val/" + var,
                mean_losses[var],
                sync_dist=True
            )

    def test_step(self, batch: Any, batch_idx: int):
        x, y, lead_times = batch
        _, pred = self.net.forward(
            x,
            y,
            lead_times,
            self.categories['input'],
            self.categories['output'],
            metric=None,
            lat=self.lat,
        )
        results = eval_loss(pred, y, lead_times, self.loss, self.lead_times, phase='test', target_v=self.categories['output'][0], normalizer=self.normalizer)

        self.test_step_outputs.append(results)
        return results

    def on_test_epoch_end(self) -> None:
        node_loss = collect_outputs(self.test_step_outputs, False)
        self.test_step_outputs.clear()  # free memory

        if isinstance(node_loss, list):
            node_loss = node_loss[0]
    
        all_losses = self.all_gather(node_loss)
        mean_losses = {k: float(torch.mean(x)) for k, x in all_losses.items()}

        # log mean losses
        for var in mean_losses.keys():
            self.log(
                "test/" + var,
                mean_losses[var],
                sync_dist=True
            )
        
        # Save evaluation results
        results_path = Path(f'./results/{self.version}_results.json')
        
        with open(results_path, 'w') as fp:
            json.dump(mean_losses, fp, indent=4)

        fp.close()
        

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        optimizer = adam.FusedAdam(
            [
                {
                    "params": decay,
                    "lr": hparams['lr'],
                    "betas": (hparams['beta_1'], hparams['beta_2']),
                    "weight_decay": hparams['weight_decay'],
                },
                {
                    "params": no_decay,
                    "lr": hparams['lr'],
                    "betas": (hparams['beta_1'], hparams['beta_2']),
                    "weight_decay": 0
                },
            ]
        )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            hparams['warmup_epochs'],
            hparams['max_epochs'],
            hparams['warmup_start_lr'],
            hparams['eta_min'],
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# ## Initialize the forecast module

# In[12]:


model = RainForecastModule(net, lat2d, datamodule.normalizer)


# # Training
# 
# Finally, we will train the model using the datamodule and forecast module defined above. 

# In[13]:


from pathlib import Path

# define logger
Path(hparams['log_path']).mkdir(parents=True, exist_ok=True)
logger = loggers.TensorBoardLogger(hparams['log_path'], version=hparams['version'])
logger.log_hyperparams(params=hparams)


# In[14]:


trainer = Trainer(
    accelerator='gpu',
    devices=hparams['gpus'],
    logger=logger,
    max_epochs=hparams['max_epochs'],
    precision=16 if hparams['use_amp'] else 32,
    default_root_dir=hparams['log_path'],
    strategy=hparams['strategy'],
    callbacks=[
        EarlyStopping('val/val_loss', patience=5), 
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            dirpath='{}/{}/'.format(hparams['log_path'], hparams['version']),
            filename='epoch-{epoch:03d}',
            monitor='val/val_loss',
            save_top_k=1,
            mode='min',
            save_last=True,
            verbose=False,
            auto_insert_metric_name=False,
        )
    ],
    accumulate_grad_batches=hparams['acc_grad'],
)
torch.set_float32_matmul_precision('medium')


# In[15]:


# trainer.fit(model, datamodule=datamodule)


# ## Evaluation

# In[ ]:


trainer.test(model.cuda(), ckpt_path='/home/allen/ckpts/pyrain-finetune-template/last.ckpt', datamodule=datamodule) 


# We can compare results to the baselines from the [Rainbench](https://arxiv.org/abs/2012.09670) paper:
# <div>
# <img src="images/baseline.png" width="500"/>
# </div>

# In[ ]:




