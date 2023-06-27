from loftr import LoFTR
import torch

device = torch.device('cuda')
model = LoFTR(pretrained='indoor_new')
model.eval().to(device)


with torch.no_grad():
    dummy_image = torch.randn(1, 1, 480, 640, device=device)
    input_dict = {
        "image0": dummy_image,
        "image1": dummy_image,
    }
    torch.onnx.export(model, (dummy_image, dummy_image), 'onnx_models/loftr_indoor_ds_new.onnx', input_names=['image0', 'image1'], output_names=['mkpts0'],verbose=True, opset_version=11)

# model = onnx.load('loftr_indoor_ds_new.onnx')
# onnx.checker.check_model(model)