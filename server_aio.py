# Python std lib
from time import perf_counter
import asyncio

# 3rd party libs
import grpc
from simber import Logger
import grpc_config.change_det_pb2 as change_det_pb2
import grpc_config.change_det_pb2_grpc as change_det_pb2_grpc

from LoFTR_demo import inference

import torch
import onnxruntime as ort
import numpy as np
import os

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'


LOG_FORMAT = "{levelname} [{filename}:{lineno}]:"
LOG_LEVEL: str = "INFO"
logger = Logger(__name__, log_path="./logs/server.log", level=LOG_LEVEL)
logger.update_format(LOG_FORMAT)

def change_detect_from_pair_image(target: bytes, base: bytes, matcher, recognizer):
    start = perf_counter()
    boxes = inference(target, base, matcher, recognizer)
    print("time: {}".format(perf_counter() - start))
    
    return [change_det_pb2.Rect(left=l, top=t, right=r, bottom=b, label=label, logit=logit) for l, t, r, b, label, logit in boxes]

# providers = [
#     ('TensorrtExecutionProvider', {
#         'device_id': 0,
#         'trt_max_workspace_size': 4294967296,
#         'trt_dla_enable': True,
#         # 'trt_fp16_enable': True,
#     }),
#     ('CUDAExecutionProvider', {
#         'device_id': 0,
#         'arena_extend_strategy': 'kNextPowerOfTwo',
#         'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
#         'cudnn_conv_algo_search': 'EXHAUSTIVE',
#         'do_copy_in_default_stream': True,
#     })
# ]

onnx_root = 'onnx_models/'
providers = ['CUDAExecutionProvider', 'TensorrtExecutionProvider']

class DetService(change_det_pb2_grpc.DetServicer):
    def __init__(self) -> None:
        super().__init__()
        chg_det_model = 'loftr_indoor_ds_new_infer.sim.onnx'
        osr_model = "osr_indoor_crl_infer.onnx"
        session = ort.InferenceSession(onnx_root + chg_det_model, providers=providers)
        cls_session = ort.InferenceSession(onnx_root + osr_model, providers=providers)
        self.matcher = session
        self.recognizer = cls_session

        # warmup
        dummpy_input = np.random.randn(1, 1, 480, 640).astype(np.float32)
        dummpy_input2 = np.random.randn(1, 3, 224, 224).astype(np.float32)
        self.matcher.run(None, {"image0": dummpy_input,  
              "image1": dummpy_input})
        self.recognizer.run(None, {'input': dummpy_input2})

    def Detect(self, request: change_det_pb2.DetInput, context):
        return change_det_pb2.DetResult(boxes=change_detect_from_pair_image(request.target, request.base, self.matcher, self.recognizer))


async def serve():
    server = grpc.aio.server()
    change_det_pb2_grpc.add_DetServicer_to_server(DetService(), server)
    # using ip v6
    adddress = "[::]:13000"
    server.add_insecure_port(adddress)
    logger.info(f"[📡] Starting server on {adddress}")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())