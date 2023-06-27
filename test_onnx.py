import onnxruntime as ort
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import kornia as K
import numpy as np
import torch
import operator
from sklearn.cluster import DBSCAN
from collections import defaultdict
import matplotlib.pyplot as plt
from loftr import LoFTR
from time import perf_counter


device = torch.device('cuda')

sess_options = ort.SessionOptions()
# sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
# sess_options.optimized_model_filepath = 'loftr_indoor_ds_new_optimized.onnx'
# model_path = 'loftr_indoor_ds_new_shape_inference.onnx'
model_path = 'onnx_models/loftr_indoor_ds_new_infer.onnx'

# session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'TensorrtExecutionProvider'], sess_options=sess_options)
session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])


print(session.get_providers())

def load_torch_image(fname):
    img = K.image_to_tensor(cv2.imread(fname), False).float() / 255.
    img = K.color.bgr_to_rgb(img)
    return img

image_dir = 'images/'

fname1 = image_dir + 'target2.jpg'
fname2 = image_dir + 'base2.jpg'

img1 = K.geometry.resize(load_torch_image(fname1), (480, 640), antialias=True)
img2 = K.geometry.resize(load_torch_image(fname2), (480, 640), antialias=True)

img1_ori = cv2.resize(cv2.imread(fname1), (640, 480))
img2_ori = cv2.resize(cv2.imread(fname2), (640, 480))

input_dict = {"image0": K.color.rgb_to_grayscale(img1).numpy(), # LofTR works on grayscale images only 
              "image1": K.color.rgb_to_grayscale(img2).numpy()}

start = perf_counter()
outputs = session.run(['mkpts0'], input_dict)
print(perf_counter() - start)

mkpts0 = outputs[0]
mkpts0 = sorted(mkpts0, key = operator.itemgetter(1,0))

def find_rect(pts):
    left, top, right, bottom = 640, 480, 0, 0
    for pt in pts:
        left = pt[0] if pt[0] < left else left
        top = pt[1] if pt[1] < top else top
        right = pt[0] if pt[0] > right else right
        bottom = pt[1] if pt[1] > bottom else bottom

    return int(left), int(top),  int(right), int(bottom)



def draw_features(image, features, img_size, color, draw_text=False):
  indices = range(len(features))
  sx = image.shape[1] / img_size[0]
  sy = image.shape[0] / img_size[1]

  for i, point in zip(indices, features):
    point_int = (int(round(point[0] * sx)), int(round(point[1] * sy)))
    cv2.circle(image, point_int, 1, color, -1, lineType=16)
    if draw_text:
      cv2.putText(image, str(i), point_int, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

# draw_features(img1_ori, mkpts0, (640, 480), color=(0, 255, 0))
# draw_features(img2_ori, mkpts1, (640, 480), color=(0, 255, 255))

left, top, right, bottom = find_rect(mkpts0)

cv2.rectangle(img1_ori, (left, top), (right, bottom), (0, 255, 0), 2)


# pts = set([(i, j) for i, j in mkpts0])

def expand_image(region, image_size):
    l, t, r, b = region
    w, h = image_size

    l -= 20
    r += 20
    t -= 15
    b += 15

    l = max(0, l)
    r = min(w, r)
    t = max(0, t)
    b = min(h, b)

    return l, t, r, b

m = 0
res = []
for row in range(top, bottom+1, 8):
    for col in range(left, right+1, 8):
        if m < len(mkpts0):
            if row == mkpts0[m][1] and col == mkpts0[m][0]:
                m += 1
            elif row != top and row != bottom and col != left and col != right:
                res.append([col, row])
        elif row != top and row != bottom and col != left and col != right:
            res.append([col, row])

# draw_features(img1_ori, res, (640, 480), color=(255, 0, 0))

clustering = DBSCAN(eps=8, min_samples=5, metric='manhattan').fit(np.array(res))

block = defaultdict(list)
for idx, val in enumerate(clustering.labels_):
    if val > 0:
        block[val].append(res[idx])

for pts in block.values():
    if len(pts) > 20:
        l, t, r, b = find_rect(pts)
        ratio = (r - l) / (b - t)
        wiz = ratio + 1/ratio
        print(wiz)
        if wiz < 4:
            draw_features(img1_ori, pts, (640, 480), color=(255, 0, 0))
            cv2.rectangle(img1_ori, (l, t), (r, b), (0, 0, 0), 2)
            l, t, r, b = expand_image((l, t, r, b), (640, 480))
            cv2.rectangle(img1_ori, (l, t), (r, b), (0, 255, 0), 2)


# combine images
res_img = np.hstack((img1_ori, img2_ori))
res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize = (20, 10))
plt.imshow(res_img)
plt.savefig("result.png")
# plt.show()