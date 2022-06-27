# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
import torch
import torchvision
import torch.nn as nn

#model = torchvision.models.mobilenet_v3_small(pretrained=True)

model = torchvision.models.mobilenet_v3_large()

last_channel = 1280
lastconv_output_channels = 960

model.classifier = nn.Sequential(
    nn.Linear(lastconv_output_channels, last_channel),
    nn.Hardswish(inplace=True),
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(last_channel, 3),
)

model.load_state_dict(torch.load("mobilenet_v3_large_finetuned.pt"))
model.eval()


batch_size = 1

x = torch.randn(batch_size, 3, 520, 520, requires_grad=True)
torch_out = model(x)

torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "mobilenet_v3_finetuned.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})