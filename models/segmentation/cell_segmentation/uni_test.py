import sys

sys.path.append("/data/yangyang/bioLLMs/CellViT_Adapter_UNI/")

import torch

from cellvit import CellViTUNI
from vit_timm import TimmVisionTransformer


model = CellViTUNI(num_nuclei_classes=6, num_tissue_classes=1)

# model2 = TimmVisionTransformer(img_size=224, patch_size=16, init_values=1e-5, dynamic_img_size=True, 
#                                embed_dim=1024, depth=24, num_heads=16, num_classes=1)

a = torch.ones((2,3,224,224))

# x = model(a)
ckpt_path = "/data/yangyang/bioLLMs/UNI/pretrained_models/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin"
state_dict = torch.load(ckpt_path, map_location="cpu")
# msg = model.load_state_dict(state_dict, strict=False)
