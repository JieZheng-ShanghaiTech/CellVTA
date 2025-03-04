from vit_timm import TimmVisionTransformer


model = TimmVisionTransformer(img_size=224, patch_size=16, init_values=1e-5, dynamic_img_size=True, num_classes=1)





