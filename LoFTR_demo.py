import cv2
import kornia as K
import numpy as np
import torch

import operator
from sklearn.cluster import DBSCAN
from collections import defaultdict
import torchvision.transforms as transforms
from PIL import Image

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

def input_trans(img_bytes):
    """
    Args:
        img_bytes: bytes of image in jpg format 
    """
    img_arr = np.frombuffer(img_bytes, np.uint8)
    img_ori = cv2.imdecode(img_arr, -1)
    img_tensor = K.image_to_tensor(img_ori, False).float() / 255.
    img = K.color.bgr_to_rgb(img_tensor)
    img_resized = K.geometry.resize(img, (480, 640), antialias=True)

    img = K.color.rgb_to_grayscale(img_resized)

    return img, img_resized

def find_rect(pts):
    """
    Args:
        pts: list of [x, y]
    """
    left, top, right, bottom = 640, 480, 0, 0
    for pt in pts:
        left = pt[0] if pt[0] < left else left
        top = pt[1] if pt[1] < top else top
        right = pt[0] if pt[0] > right else right
        bottom = pt[1] if pt[1] > bottom else bottom

    return int(left), int(top),  int(right), int(bottom)

def expand_image(region, image_size):
    l, t, r, b = region
    w, h = image_size

    l -= 15
    r += 15
    t -= 15
    b += 15

    l = max(0, l)
    r = min(w, r)
    t = max(0, t)
    b = min(h, b)

    return l, t, r, b

@torch.inference_mode()
def inference(target_bytes, base_bytes, matcher, recognizer):
    target_trans, target_resized = input_trans(target_bytes)
    base_trans, _ = input_trans(base_bytes)
    input_dict = {"image0": target_trans.numpy(),  
              "image1": base_trans.numpy()}
    correspondense = matcher.run(None, input_dict)
    mkpts0 = correspondense[0]
    mkpts0 = sorted(mkpts0, key = operator.itemgetter(1,0))
    left, top, right, bottom = find_rect(mkpts0)

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

    clustering = DBSCAN(eps=8, min_samples=5, metric='manhattan').fit(np.array(res))
    block = defaultdict(list)

    for idx, val in enumerate(clustering.labels_):
        if val > 0:
            block[val].append(res[idx])

    boxes = []
    for pts in block.values():
        if len(pts) > 20:
            l, t, r, b = find_rect(pts)
            ratio = (r - l) / (b - t)
            wiz = ratio + 1/ratio
            if wiz < 4:
                l, t, r, b = expand_image((l, t, r, b), (right-left, bottom-top))
                # target_ori = cv2.resize(target_ori, (640, 480))
                # cropped_img = target_ori[t:b, l:r]
                # cv2.imwrite('cropped_img.jpg', cropped_img)
                # cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                # cropped_img = test_transform(Image.fromarray(cropped_img))
                # cropped_img = cropped_img.unsqueeze(0)
                cropped_img = K.geometry.crop_and_resize(target_resized, torch.tensor([[[l, t], [r, t], [r, b], [l, b]]]), (224, 224))
                output = recognizer.run(None, {'input': cropped_img.numpy()})
                label = int(output[0])
                logit = np.max(output[1])
                boxes.append([l/640, t/480, r/640, b/480, label, logit])

    boxes = list(sorted(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True))

    return boxes