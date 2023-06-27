# Python std lib
import datetime
import os
import socket
import time
import sys
import contextlib
from concurrent import futures
import pickle
from typing import List
from time import perf_counter

# 3rd party libs
import cv2
import grpc
from simber import Logger
import grpc_config.change_det_pb2 as change_det_pb2
import grpc_config.change_det_pb2_grpc as change_det_pb2_grpc

from LoFTR_demo import inference
# from loftr import LoFTR
import torch
import kornia.feature as KF
import onnxruntime as ort
import asyncio


LOG_FORMAT = "{levelname} [{filename}:{lineno}]:"
LOG_LEVEL: str = "INFO"
logger = Logger(__name__, log_path="./logs/server.log", level=LOG_LEVEL)
logger.update_format(LOG_FORMAT)

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 2))

def change_detect_from_pair_image(target: bytes, base: bytes, matcher, device):
    start = perf_counter()
    boxes = inference(target, base, matcher, device)
    print("time: {}".format(perf_counter() - start))
    
    return [change_det_pb2.Rect(left=l, top=t, right=r, bottom=b) for l, t, r, b in boxes]


# class DetService(change_det_pb2_grpc.DetServicer):
#     def __init__(self, idx) -> None:
#         super().__init__()
#         device = torch.device(idx)
#         self.matcher = KF.LoFTR(pretrained='indoor_new') #LoFTR(pretrained='indoor_new')
#         self.matcher.eval().to(device)
#         self.matcher.coarse_matching.thr = 0.1
#         self.device = device

#     def Detect(self, request: change_det_pb2.DetInput, context):
#         return change_det_pb2.DetResult(boxes=change_detect_from_pair_image(request.target, request.base, self.matcher, self.device))

class DetService(change_det_pb2_grpc.DetServicer):
    def __init__(self, idx) -> None:
        super().__init__()
        device = torch.device('cuda')
        model_path = 'onnx_models/loftr_indoor_ds_new_infer.onnx'
        session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'TensorrtExecutionProvider'])
        self.matcher = session
        # self.matcher = LoFTR(pretrained='indoor_new') #LoFTR(pretrained='indoor_new')
        # self.matcher.eval().to(device)
        # self.matcher.coarse_matching.thr = 0.1
        self.device = device

    def Detect(self, request: change_det_pb2.DetInput, context):
        return change_det_pb2.DetResult(boxes=change_detect_from_pair_image(request.target, request.base, self.matcher, self.device))

@contextlib.contextmanager
def _reserve_port():
    """Find and reserve a port for all subprocesses to use"""
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    if sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 0:
        raise RuntimeError("Failed to set SO_REUSEPORT.")
    sock.bind(("", 13000))
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()

def _run_server(bind_address, idx):
    async def serve():
        # logger.info("idx: {}".format(idx))
        # torch.cuda.set_device(torch.device(idx))
        logger.info(f"Server started. Awaiting jobs...")
        server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=1),
            options=[
                ("grpc.max_send_message_length", -1),
                ("grpc.max_receive_message_length", -1),
                ("grpc.so_reuseport", 1),
                ("grpc.use_local_subchannel_pool", 1),
            ],
        )
        change_det_pb2_grpc.add_DetServicer_to_server(DetService(idx), server)
        server.add_insecure_port(bind_address)
        await server.start()
        await server.wait_for_termination()

    asyncio.run(serve())

def main():
    """
    Inspired from https://github.com/grpc/grpc/blob/master/examples/python/multiprocessing/server.py
    """
    logger.info(f"Initializing server with {NUM_WORKERS} workers")
    with _reserve_port() as port:
        bind_address = f"0.0.0.0:{port}"
        logger.info(f"Binding to {bind_address}")
        sys.stdout.flush()
        workers : List[torch.multiprocessing.Process]= []
        for idx in range(NUM_WORKERS):
            worker = torch.multiprocessing.Process(target=_run_server, args=(bind_address, idx))
            worker.start()
            workers.append(worker)
        for worker in workers:
            worker.join()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
