import onnx
from onnxconverter_common import float16
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="path/to/model.onnx")
parser.add_argument("--output_path", type=str,
                    default="path/to/model_fp16.onnx")
args = parser.parse_args()

model = onnx.load(args.model_path)
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, args.output_path)
