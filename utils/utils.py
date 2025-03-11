import csv
import json

import onnx
import torch
from thop import clever_format, profile


# json file to csv
def json2csv(json_file, csv_file):
    with open(json_file) as f:
        data = json.load(f)
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data[0].keys())
        for row in data:
            writer.writerow(row.values())


def count_parameters(model, input_size=(1, 3, 224, 224)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dummy_input = torch.randn(input_size).to(device)
    macs, params = profile(model, (dummy_input, ), verbose=False)
    #-------------------------------------------------------------------------------#
    #   flops * 2 because profile does not consider convolution as two operations.
    #-------------------------------------------------------------------------------#
    flops         = macs * 2
    # flops, params = clever_format([flops, params], "%.2f ")
    macs, flops, params = clever_format([macs, flops, params], "%.2f ")
    print(f'Total MACs:   {macs}')
    print(f'Total GFLOPs: {flops}')
    print(f'Total params: {params}')
    return macs, flops, params


def export2onnx(model, input_size=(1, 3, 224, 224), model_name="mobilenetv4_small.onnx"):
    model.eval()
    x = torch.randn(input_size)
    torch.onnx.export(model, x,
        model_name,
        verbose=False,
        input_names=["input"],
        output_names=["output"],
        opset_version=20,
    )


def simplify_onnx(original_model, simplified_model):
    from onnxsim import simplify
    model = onnx.load(original_model)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, simplified_model)


if __name__ == '__main__':
    json2csv('/Users/haoyu/Documents/Projects/xfeat_lightglue_onnx/onnxruntime_profile__2025-03-11_16-03-51.json', 'data.csv')