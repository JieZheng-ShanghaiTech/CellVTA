import sys

sys.path.append("/data/yangyang/bioLLMs/CellViT_Adapter_UNI/")

import torch
from models.segmentation.cell_segmentation.cellvit import CellViTUNIAdapter


a = torch.ones([2,3,224,224])
a = a.to("cuda:0")

model = CellViTUNIAdapter(
                        #img_size=224, patch_size=16, init_values=1e-5, dynamic_img_size=True, 
                        #embed_dim=1024, depth=24, num_heads=16, num_classes=1,
                        num_nuclei_classes=5,
                        num_tissue_classes=1,
                        drop_rate=0,
                        conv_inplane=64, 
                        n_points=4,
                        deform_num_heads=8, 
                        # mlp_ratio=4,
                        drop_path_rate=0.4,
                        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
                        with_cffn=False,
                        cffn_ratio=0.25, 
                        deform_ratio=0.5, 
                        add_vit_feature=True)

# model = model.to("cuda:0")

# x = model(a)