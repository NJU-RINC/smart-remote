# 边缘端

## 环境配置

```shell
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 grpcio grpcio-tools opencv-python kornia onnxruntime-gpu scikit-learn einops matplotlib simber --proxy http://127.0.0.1:7890
```

```shell
python -m onnxruntime.quantization.preprocess --input loftr_indoor_ds_new.onnx --output loftr_indoor_ds_new_infer.onnx
```

```shell
python -m grpc_tools.protoc -I ./grpc_config --python_out=./grpc_config --pyi_out=./grpc_config --grpc_python_out=./grpc_config ./grpc_config/change_det.proto
```
