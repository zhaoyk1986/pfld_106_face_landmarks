import onnx
import os
import argparse
from models.mobilev3_pfld import PFLDInference
import torch
import onnxsim

parser = argparse.ArgumentParser(description='pytorch2onnx')
parser.add_argument(
    '--torch_model',
    default="/Users/xintao/Documents/GitHub/pfld_106_face_landmarks/checkpoint/v3/v3.pth")
parser.add_argument('--onnx_model', default="./output/pfld.onnx")
parser.add_argument(
    '--onnx_model_sim',
    help='Output ONNX model',
    default="./output/v3.onnx")
args = parser.parse_args()

print("=====> load pytorch checkpoint...")
plfd_backbone = PFLDInference()
plfd_backbone.eval()
plfd_backbone.load_state_dict(torch.load(args.torch_model, map_location=torch.device('cpu'))['plfd_backbone'])
print("PFLD bachbone:", plfd_backbone)

print("=====> convert pytorch model to onnx...")
dummy_input = torch.randn(1, 3, 112, 112)
input_names = ["input"]
output_names = ["output1", "output"]
torch.onnx.export(
    plfd_backbone,
    dummy_input,
    args.onnx_model,
    verbose=True,
    input_names=input_names,
    output_names=output_names)


print("====> check onnx model...")
model = onnx.load(args.onnx_model)
onnx.checker.check_model(model)

print("====> Simplifying...")
model_opt = onnxsim.simplify(args.onnx_model)
# print("model_opt", model_opt)
onnx.save(model_opt, args.onnx_model_sim)
print("onnx model simplify Ok!")
