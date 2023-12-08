import torch

from climax.arch import ClimaX
from climax.new_arch import NewClimaX
from climax.utils.loader_utils import AVAILABLE_CMIP6_CKPTS, DEFAULT_VAR_CLIMAX_V1, DEFAULT_VAR_CLIMAX_V2

def load_climax(
    version: int = 1,
    size: str = 'small',
    pretraining_data: str = 'cmip6',
    resolution: str = 5.625,
    preset_net: torch.nn.Module = None
):
    """
    Load pretrained ClimaX based on model version, size, pretraining data, resolution
    version: ClimaX version, available versions -- 1, 2
    size: size of the backbone, only applicable to version 2 -- 'small', 'base', or 'large'
    pertraining_data: the data the model was pretrained on -- 'vision' or 'cmip6', 'vision' is only applicable to version 2
    resolution: resolution of data the model was pretrained on -- 5.625 or 1.40625, version 2 is currently available at 1.40625 only
    net (optional): a custom network based on climax, ignore this if using the original architecture

    Note that some combinations may not be available. Check the available checkpoints by looking at AVAILABLE_CMIP6_CKPTS
    """
    if preset_net is not None:
        net = preset_net
    else: ### initialize network based on ckpt_name
        if version == 1:
            print (f'Initializing ClimaX-v1 at resolution {resolution}deg.')
            net = ClimaX(
                default_vars=DEFAULT_VAR_CLIMAX_V1,
                img_size=[5, 12] if resolution==5.625 else [128, 256],
                patch_size=2 if resolution==5.625 else 4,
                embed_dim=1024,
                depth=8,
                decoder_depth=2,
                num_heads=16,
                mlp_ratio=4.0,
                drop_path=0.1,
                drop_rate=0.1,
                parallel_patch_embed=False,
            )
        else:
            print (f'Initializing ClimaX-v2 {size} at resolution {resolution}deg.')
            if size == 'small':
                backbone = 'dinov2_vits14'
            elif size == 'base':
                backbone = 'dinov2_vitb14'
            else:
                backbone = 'dinov2_vitl14'
            net = NewClimaX(
                default_vars=DEFAULT_VAR_CLIMAX_V2,
                img_size=[5, 12] if resolution==5.625 else [128, 256],
                patch_size=2 if resolution==5.625 else 4,
                backbone=backbone,
                vision_pretrained=True if pretraining_data == 'vision' else False,
                decoder_depth=2,
                parallel_patch_embed=True
            )

    if pretraining_data == 'cmip6': ### load cmip6 pretrained weights
        if version == 1:
            ckpt_url = AVAILABLE_CMIP6_CKPTS[f'climaxv{version}-{resolution}']
            print ('Loading CMIP6 pretrained checkpoint from', ckpt_url)
            ckpt = torch.hub.load_state_dict_from_url(ckpt_url, map_location='cpu')
            state_dict = ckpt['state_dict']
            state_dict = {k[4:]: v for k, v in state_dict.items()}

            for k in list(state_dict.keys()):
                if "channel" in k:
                    state_dict[k.replace("channel", "var")] = state_dict[k]
                    del state_dict[k]

            for k in list(state_dict.keys()):
                if k not in net.state_dict().keys() or state_dict[k].shape != net.state_dict()[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del state_dict[k]
            
            msg = net.load_state_dict(state_dict, strict=False)
            print (msg)
        else:
            ckpt_url = AVAILABLE_CMIP6_CKPTS[f'climaxv{version}-{size}-{resolution}']
            print ('Loading CMIP6 pretrained checkpoint from', ckpt_url)
            ckpt = torch.hub.load_state_dict_from_url(ckpt_url, map_location='cpu')
            state_dict = ckpt['state_dict']
            state_dict = {k[4:]: v for k, v in state_dict.items()}

            for k in list(state_dict.keys()):
                if "pretrained_backbone" in k:
                    state_dict[k.replace("pretrained_backbone", "backbone")] = state_dict[k]
                    del state_dict[k]

            for k in list(state_dict.keys()):
                if "channel" in k:
                    state_dict[k.replace("channel", "var")] = state_dict[k]
                    del state_dict[k]
            
            for k in list(state_dict.keys()):
                if "embedding." in k:
                    state_dict[k.replace("embedding.", "")] = state_dict[k]
                    del state_dict[k]

            for k in list(state_dict.keys()):
                if k not in net.state_dict().keys() or state_dict[k].shape != net.state_dict()[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del state_dict[k]

            msg = net.load_state_dict(state_dict, strict=False)
            print (msg)

    return net

# model = load_climax(
#     version = 2,
#     size='large',
#     pretraining_data='cmip6',
#     resolution=1.40625
# )