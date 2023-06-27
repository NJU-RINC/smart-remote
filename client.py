from grpc_config.change_det_pb2_grpc import DetStub
from grpc_config.change_det_pb2 import DetInput, DetResult
import grpc
import cv2
from time import perf_counter
import asyncio

image_dir = 'images/'

fname1 = image_dir + 'target1.jpg'
fname2 = image_dir + 'base1.jpg'


target = cv2.imread(fname1)
base = cv2.imread(fname2)

_, target_encode = cv2.imencode('.jpg', target)
target_bytes = target_encode.tobytes()
_, base_encode = cv2.imencode('.jpg', base)
base_bytes = base_encode.tobytes()

async def main():
    async with grpc.aio.insecure_channel("localhost:13000") as channel:
        stub = DetStub(channel)
        print("conneced")
        for _ in range(20):
            start = perf_counter()
            res: DetResult = await stub.Detect(DetInput(target=target_bytes, base=base_bytes))
            for box in res.boxes:
                left, top, right, bottom, label = box.left, box.top, box.right, box.bottom, box.label
                print(left, top, right, bottom, label)
            print("time: {}".format(perf_counter() - start))

# def main():
#      with grpc.insecure_channel("localhost:13000") as channel:
#         stub = DetStub(channel)
#         print("conneced")
#         for _ in range(100):
#             start = perf_counter()
#             res: DetResult = stub.Detect(DetInput(target=target_bytes, base=base_bytes))
#             for box in res.boxes:
#                 left, top, right, bottom = box.left, box.top, box.right, box.bottom
#                 print(left, top, right, bottom)
#             print("time: {}".format(perf_counter() - start))



if __name__ == "__main__":
    asyncio.run(main())
    # main()