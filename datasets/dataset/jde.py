import glob
import math
import os
import os.path as osp
import random
import time
from collections import OrderedDict

import cv2
import json
import numpy as np
import torch
import copy

from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
from cython_bbox import bbox_overlaps as bbox_ious
from opts import opts
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.utils import xyxy2xywh, generate_anchors, xywh2xyxy, encode_delta
from PIL import Image, ImageDraw


class LoadImages:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        if os.path.isdir(path):
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            self.files = sorted(glob.glob('%s/*.*' % path))
            self.files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, self.files))
        elif os.path.isfile(path):
            self.files = [path]

        self.nF = len(self.files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration
        img_path = self.files[self.count]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return img_path, img, img0

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0

    def __len__(self):
        return self.nF  # number of files


class LoadVideo:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        self.w, self.h = 1920, 1080
        print('Lenth of the video: {:d} frames'.format(self.vn))

    def get_size(self, vw, vh, dw, dh):
        wa, ha = float(dw) / vw, float(dh) / vh
        a = min(wa, ha)
        return int(vw * a), int(vh * a)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration
        # Read image
        res, img0 = self.cap.read()  # BGR
        assert img0 is not None, 'Failed to load frame {:d}'.format(self.count)
        img0 = cv2.resize(img0, (self.w, self.h))

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return self.count, img, img0

    def __len__(self):
        return self.vn  # number of files


class LoadImagesAndLabels:  # for training
    def __init__(self, path, img_size=(1088, 608), augment=False, transforms=None):
        with open(path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [x.replace('\n', '') for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                            for x in self.img_files]

        self.nF = len(self.img_files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms

    def __getitem__(self, files_index):
        img_path = self.img_files[files_index]
        label_path = self.label_files[files_index]
        return self.get_data(img_path, label_path)

    def get_data(self, img_path, label_path, box_aug):
        height = self.height
        width = self.width
        img = cv2.imread(img_path)  # BGR
        if img is None:
            raise ValueError('File corrupt {}'.format(img_path))
        augment_hsv = True
        if self.augment and augment_hsv: # 颜色转换 BGR -> HSV
            # SV augmentation by 50%
            fraction = 0.50
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)

            a = (random.random() * 2 - 1) * fraction + 1
            S *= a
            if a > 1:
                np.clip(S, a_min=0, a_max=255, out=S)

            a = (random.random() * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, a_min=0, a_max=255, out=V)

            img_hsv[:, :, 1] = S.astype(np.uint8)
            img_hsv[:, :, 2] = V.astype(np.uint8)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        h, w, _ = img.shape
        img, ratio, padw, padh = letterbox(img, height=height, width=width) # 将图片缩放到统一大小（1088*608）

        # Load labels
        if os.path.isfile(label_path): # cxcywh ——> xyxy
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 2] = ratio * w * (labels0[:, 2] - labels0[:, 4] / 2) + padw
            labels[:, 3] = ratio * h * (labels0[:, 3] - labels0[:, 5] / 2) + padh
            labels[:, 4] = ratio * w * (labels0[:, 2] + labels0[:, 4] / 2) + padw
            labels[:, 5] = ratio * h * (labels0[:, 3] + labels0[:, 5] / 2) + padh
        else:
            labels = np.array([])

        # Augment image and labels
        if self.augment: # 数据增强 # 这一步，label的顺序会发生变化（有一些目标框会被删掉），需要修改（2023.2.6）
            img, labels, M = random_affine(img, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.50, 1.20))
            # 2023.03.20 box层面的数据增强
            if box_aug == 1:
                img = aug_box(img, labels)
            # vis(img, box=labels[0][2:6])

        plotFlag = False
        if plotFlag:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(50, 50))
            plt.imshow(img[:, :, ::-1])
            plt.plot(labels[:, [1, 3, 3, 1, 1]].T, labels[:, [2, 2, 4, 4, 2]].T, '.-')
            plt.axis('off')
            plt.savefig('test.jpg')
            time.sleep(10)

        nL = len(labels)
        if nL > 0:
            # convert xyxy to xywh
            labels[:, 2:6] = xyxy2xywh(labels[:, 2:6].copy())  # / height
            labels[:, 2] /= width # 相对值（0~1之间）
            labels[:, 3] /= height
            labels[:, 4] /= width
            labels[:, 5] /= height
        if self.augment:
            # random left-right flip 随机左右翻转
            lr_flip = True
            if lr_flip & (random.random() > 0.5):
                img = np.fliplr(img)
                if nL > 0:
                    labels[:, 2] = 1 - labels[:, 2] # 这一步，label的顺序是没有变的

        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB

        if self.transforms is not None:
            img = self.transforms(img) # 在这个代码中这里不仅仅是ToTensor;
            # 还改变了值，从整数变到0-1之间；
            # 还改变了形状，从（608，1088，3）变为（3，608，1088）
            # ToTensor()，转换一个PIL库的图片或者numpy的数组为tensor张量类型；转换从[0, 255]->[0, 1]
            # 实现原理，即针对不同类型进行处理，原理即各值除以255，最后通过torch.from_numpy将PIL Image or numpy.ndarray
            # 针对具体数值类型比如Int32,int16,float等转成torch.tensor数据类型
            # 在转换过程中有一个轴的转置操作transpose(2, 0, 1) 和contiguous() 函数

        return img, labels, img_path, (h, w)

    def __len__(self):
        return self.nF  # number of batches


def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


def random_affine(img, targets=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    height = img.shape[0]
    width = img.shape[1]

    # Rotation and Scale 旋转和缩放
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s) # 绕中心点按照角度旋转并按比例缩放

    # Translation 平移
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear 错切（错切变换实际上是平面景物在投影平面上的非垂直投影效果）
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also 根据前面对图片的增强处理标签，使其相对应
    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 2:6].copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            #np.clip(xy[:, 0], 0, width, out=xy[:, 0])
            #np.clip(xy[:, 2], 0, width, out=xy[:, 2])
            #np.clip(xy[:, 1], 0, height, out=xy[:, 1])
            #np.clip(xy[:, 3], 0, height, out=xy[:, 3])
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))

            # 2023.2.6 以下操作会导致标签顺序变化，需要修改
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10) # 变换后的有效目标框为TRUE
            # 宽>4，高>4， 变换后的面积/原始面积 > 0.1， 最大宽高比<10

            # 这一步会导致目标顺序变化（会删掉变换后不符合条件的目标框）
            # targets = targets[i]   # 会有筛选
            # targets[:, 2:6] = xy[i]
            # # 目标框在图片范围内
            # targets = targets[targets[:, 2] < width] # width=1088
            # targets = targets[targets[:, 4] > 0]
            # targets = targets[targets[:, 3] < height] # height=608
            # targets = targets[targets[:, 5] > 0]

            # 2023.2.6 暂时保留无效的目标框
            targets[:, 2:6] = xy
            # 增加一个标记位，标记是否为有效框
            tag = i & (targets[:, 2] < width) & (targets[:, 4] > 0) & (targets[:, 3] < height) & (targets[:, 5] > 0)
            targets_tag = np.zeros((n, 7))
            targets_tag[:, :6] = targets # 前6位为原本的标签信息
            targets_tag[:, 6] = tag # 最有一位为box有效标记信息，1为有效，0为无效; 后续用reg_mask标记有效框
            targets = targets_tag # 最后输出的标签信息是带有有效标记的

        return imw, targets, M
    else:
        return imw

# 2023.03.20 box层面的数据增强
# 根据SimCLR，选择其中比较贴合我们任务的几种：
# Crop and resize 随机剪裁并调整大小（随机裁剪之后再resize成原来的大小，模拟目标从远处走近）
#   （ Crop, resize (and flip) 随机剪裁并调整大小、翻转）
# Cutout 部分遮挡。
#           CUtout与RandomErasing类似，也是通过填充区域，从而将填充区域的图像信息遮挡，有利于提高模型的泛化能力。
#           与random Erasing不同的是，Cutout使用固定大小的正方形区域，采用全0填充，而且允许正方形区域在图片外
#           （由于这点，Cutout非正方形区域都在边界处）。
def aug_box(img, labels):
    box = labels[:, 2:6]
    box_num = box.shape[0]
    for i in range(box_num): # 每个box随机进行增强变换
        r = random.randint(0, 1) # 根据该随机数选择增强类型
        if r == 0: # 不增强
            continue
        elif r == 1:
            img = box_noise(img, box[i]) # 将box中的随机一小块以一定的比例替换为随机噪声，以一定比例保留原始的像素信息
        elif r == 2:
            img = box_patch(img, box[i]) # 将box中的随机一小块替换为图片中的另一个块
        elif r == 3:
            img = box_crop(img, box[i]) # 随机裁剪之后再resize成原来的大小(额外随机翻转)，模拟目标从远处走近
    # vis(img, box[0])
        # 2023.04.03 新尝试的数据增强方法
            box2 = box[random.randint(0, box_num-1)]
        elif r == 4:
            img = mixup(img, box[i], box2)
        elif r == 5:
            img = cut_out(img, box[i])
        elif r == 6:
            img = cut_mix(img, box[i], box2)
    return img

# Yolov4的mosaic数据增强参考了CutMix数据增强方式，理论上具有一定的相似性。
# CutMix数据增强方式利用两张图片进行拼接，但是mosaic利用了四张图片，
# 根据论文所说其拥有一个巨大的优点是丰富检测物体的背景，且在BN计算的时候一下子会计算四张图片的数据。
# （暂未实现 https://www.jianshu.com/p/639f9ecc1328）
# 可能和我们的任务不太符，我们一个检测box中真实的本体对象不可能以拼接的部分只在box的一个角落出现


# 2023.04.03 将一部分区域cut掉但不填充0像素而是随机填充训练集中的其他数据的区域像素值
def cut_mix(img, box1, box2): # 有点类似box_patch
    m = np.random.beta(1.0, 1.0)  # 混合比例
    # 考虑box坐标超出image边界的情况
    box1[0] = box1[0].clip(0, img.shape[1])
    box1[1] = box1[1].clip(0, img.shape[0])
    box1[2] = box1[2].clip(0, img.shape[1])
    box1[3] = box1[3].clip(0, img.shape[0])
    box_h1 = int(box1[3]) - int(box1[1])  # box 高
    box_w1 = int(box1[2]) - int(box1[0])  # box 宽
    box2[0] = box2[0].clip(0, img.shape[1])
    box2[1] = box2[1].clip(0, img.shape[0])
    box2[2] = box2[2].clip(0, img.shape[1])
    box2[3] = box2[3].clip(0, img.shape[0])
    box_h2 = int(box2[3]) - int(box2[1])  # box 高
    box_w2 = int(box2[2]) - int(box2[0])  # box 宽
    # image内的box无效的情况
    if box_w1 == 0 or box_h1 == 0 or box_w2 == 0 or box_h2 == 0:
        return img
    box_img = Image.fromarray(img)  # numpy——>Image
    box1_img = box_img.crop((box1))
    box2_img = box_img.crop((box2))
    box1_img = box1_img.resize((box_w1, box_h1))
    box2_img = box2_img.resize((box_w1, box_h1))
    cut_rat = np.sqrt(1. - m)
    cut_w = np.int(box_w1 * cut_rat) # 被替换区域的大小
    cut_h = np.int(box_h1 * cut_rat)
    cx = np.random.randint(box_w1) # 被替换区域的中心点
    cy = np.random.randint(box_h1)
    # 被替换区域的坐标(裁剪下来的box中的坐标）
    x1 = np.clip(cx - cut_w//2, 0, box_w1)
    y1 = np.clip(cy - cut_h // 2, 0, box_h1)
    x2 = np.clip(cx + cut_w // 2, 0, box_w1)
    y2 = np.clip(cx + cut_h // 2, 0, box_h1)
    box1_img = np.array(box1_img)
    box2_img = np.array(box2_img)
    box1_img[y1:y2, x1:x2, :] = box2_img[y1:y2, x1:x2, :]
    img[int(box1[1]):int(box1[3]), int(box1[0]):int(box1[2]), :] = box1_img[:, :, :]
    # vis(img, box1)
    return img

# 2023.04.03 随机的将样本中的部分区域cut掉，并且填充0像素值
def cut_out(img, box):
    m = np.float32(np.random.beta(1.0, 1.0))
    box_h = int(box[3]) - int(box[1])  # box 高
    box_w = int(box[2]) - int(box[0])  # box 宽
    x1 = random.randint(int(box[0]), int(box[2]))  # 随机选择左上顶点
    y1 = random.randint(int(box[1]), int(box[3]))
    new_w = random.randint(0, box_w)  # 随机选择宽高
    new_h = random.randint(0, box_h)
    x2 = min(int(box[2]), x1 + new_w)  # 确保随机选择的box在原来的box之内
    y2 = min(int(box[3]), y1 + new_h)
    img[y1:y2, x1:x2, 0] = 0
    img[y1:y2, x1:x2, 1] = 0
    img[y1:y2, x1:x2, 2] = 0
    # vis(img, box)
    return img

# 2023.04.03 将随机的两个box按比例混合
def mixup(img, box1, box2):
    m = np.random.beta(1.0, 1.0)  # 混合比例
    m = np.max([m, 1 - m])
    # 考虑box坐标超出image边界的情况
    box1[0] = box1[0].clip(0, img.shape[1])
    box1[1] = box1[1].clip(0, img.shape[0])
    box1[2] = box1[2].clip(0, img.shape[1])
    box1[3] = box1[3].clip(0, img.shape[0])
    box_h1 = int(box1[3]) - int(box1[1])  # box 高
    box_w1 = int(box1[2]) - int(box1[0])  # box 宽
    box2[0] = box2[0].clip(0, img.shape[1])
    box2[1] = box2[1].clip(0, img.shape[0])
    box2[2] = box2[2].clip(0, img.shape[1])
    box2[3] = box2[3].clip(0, img.shape[0])
    box_h2 = int(box2[3]) - int(box2[1])  # box 高
    box_w2 = int(box2[2]) - int(box2[0])  # box 宽
    # image内的box无效的情况
    if box_w1 == 0 or box_h1 == 0 or box_w2 == 0 or box_h2 == 0:
        return img
    box_img = Image.fromarray(img)  # numpy——>Image
    box1_img = box_img.crop((box1))
    box2_img = box_img.crop((box2))
    box1_img = box1_img.resize((box_w1, box_h1))
    box2_img = box2_img.resize((box_w1, box_h1))
    box1_img = box1_img * m + box2_img * (1-m)
    box1_img = np.array(box1_img)
    img[int(box1[1]):int(box1[3]), int(box1[0]):int(box1[2]), :] = box1_img[:, :, :]
    # vis(img, box1)
    return img

# 随机裁剪之后再resize成原来的大小(额外随机翻转)，模拟目标从远处走近。
# 改动多了之后，原图会有点面目全非，尤其是遮挡的地方，完全没有之前的信息了
def box_crop(img, box):
    # print('box:', box[0],box[1],box[2],box[3])
    # 考虑box坐标超出image边界的情况
    box[0] = box[0].clip(0, img.shape[1])
    box[1] = box[1].clip(0, img.shape[0])
    box[2] = box[2].clip(0, img.shape[1])
    box[3] = box[3].clip(0, img.shape[0])
    # print('box:', box[0],box[1],box[2],box[3])
    box_h = int(box[3]) - int(box[1])  # box 高
    box_w = int(box[2]) - int(box[0])  # box 宽
    # image内的box无效的情况
    if box_w == 0 or box_h == 0:
        return img
    x1 = random.uniform(box[0], box[0]+box_w/2)  # 随机选择左上顶点(原图顶点和中心点之间)
    y1 = random.uniform(box[1], box[1]+box_h/2)
    new_w = random.uniform(box_w/2, box_w) # 随机选择宽高;还是要保留的多一点（至少大于原box的1/2）
    new_h = random.uniform(box_h/2, box_h)
    x2 = min(box[2], x1+new_w) # 确保随机选择的box在原来的box之内
    y2 = min(box[3], y1+new_h)
    # print('crop:',x1,x2,y1,y2)
    box_img = Image.fromarray(img) # numpy——>Image
    box_img = box_img.crop((x1, y1, x2, y2))
    box_img = box_img.resize((box_w, box_h))
    flip = np.float32(np.random.beta(1.0, 1.0)) # 是否翻转（大于0.5翻转）
    if flip > 0.5:
        box_img = box_img.transpose(Image.FLIP_LEFT_RIGHT)
    box_img = np.array(box_img)
    img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :] = box_img[:, :, :]
    # vis(img, box)
    return img

# 将box中的随机一小块替换为图片中的另一个块
def box_patch(img, box):
    m = np.float32(np.random.beta(1.0, 1.0)) # 替换的比例
    # 考虑box坐标超出image边界的情况
    box[0] = box[0].clip(0, img.shape[1])
    box[1] = box[1].clip(0, img.shape[0])
    box[2] = box[2].clip(0, img.shape[1])
    box[3] = box[3].clip(0, img.shape[0])
    box_h = int(box[3]) - int(box[1])  # box 高
    box_w = int(box[2]) - int(box[0])  # box 宽
    # image内的box无效的情况
    if box_w == 0 or box_h == 0:
        return img
    x1 = random.randint(int(box[0]), int(box[2])-1)  # 随机选择左上顶点;避免取到image右下角
    y1 = random.randint(int(box[1]), int(box[3])-1)
    new_w = random.randint(0, box_w)  # 随机选择宽高
    new_h = random.randint(0, box_h)
    x2 = min(int(box[2]), x1 + new_w, img.shape[1])  # 确保随机选择的box在原来的box之内且在image之内
    y2 = min(int(box[3]), y1 + new_h, img.shape[0])
    # print('patch:',x1,x2,y1,y2)
    noise_h = y2 - y1
    noise_w = x2 - x1
    x = random.randint(0, img.shape[1]) # 在全图范围内随机选择一个点
    y = random.randint(0, img.shape[0])
    # 考虑随机选择的块超过图像边界的情况
    if x+noise_w > img.shape[1]:
        x = img.shape[1] - noise_w
    if y+noise_h > img.shape[0]:
        y = img.shape[0] - noise_h
    img[y1:y2, x1:x2, :] = (1 - m) * img[y1:y2, x1:x2, :] + m * img[y:y + noise_h, x:x + noise_w, :]  # 随机替换为该图片中的另一个块
    # vis(img, box)
    return img

# 将box中的随机一小块以一定的比例替换为随机噪声，以一定比例保留原始的像素信息（不需要考虑对齐，所以不用在乎box是否在image内）
def box_noise(img, box):
    mean = (0.4914, 0.4822, 0.4465) # Erasing value
    m = np.float32(np.random.beta(1.0, 1.0))
    box_h = int(box[3]) - int(box[1])  # box 高
    box_w = int(box[2]) - int(box[0])  # box 宽
    x1 = random.randint(int(box[0]), int(box[2]))  # 随机选择左上顶点
    y1 = random.randint(int(box[1]), int(box[3]))
    new_w = random.randint(0, box_w)  # 随机选择宽高
    new_h = random.randint(0, box_h)
    x2 = min(int(box[2]), x1 + new_w)  # 确保随机选择的box在原来的box之内
    y2 = min(int(box[3]), y1 + new_h)
    img[y1:y2, x1:x2, 0] = (1 - m) * img[y1:y2, x1:x2, 0] + m * mean[0]  # 随机噪声（保留一定原始像素信息）
    img[y1:y2, x1:x2, 1] = (1 - m) * img[y1:y2, x1:x2, 1] + m * mean[1]
    img[y1:y2, x1:x2, 2] = (1 - m) * img[y1:y2, x1:x2, 2] + m * mean[2]
    # vis(img, box)
    return img

# 2023.03.20 可视化检查
def vis(img, box):
    Img = Image.fromarray(img)
    draw = ImageDraw.Draw(Img)
    draw.rectangle([int(box[0]), int(box[1]), int(box[2]), int(box[3])], outline=(255, 0, 0))
    Img.show()


def collate_fn(batch):
    imgs, labels, paths, sizes = zip(*batch)
    batch_size = len(labels)
    imgs = torch.stack(imgs, 0)
    max_box_len = max([l.shape[0] for l in labels])
    labels = [torch.from_numpy(l) for l in labels]
    filled_labels = torch.zeros(batch_size, max_box_len, 6)
    labels_len = torch.zeros(batch_size)

    for i in range(batch_size):
        isize = labels[i].shape[0]
        if len(labels[i]) > 0:
            filled_labels[i, :isize, :] = labels[i]
        labels_len[i] = isize

    return imgs, filled_labels, paths, sizes, labels_len.unsqueeze(1)


class JointDataset(LoadImagesAndLabels):  # for training
    default_resolution = [1088, 608]
    mean = None
    std = None
    num_classes = 1

    def __init__(self, opt, root, paths, img_size=(1088, 608), augment=False, transforms=None):
        self.opt = opt
        dataset_names = paths.keys()
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        self.num_classes = 1

        for ds, path in paths.items():
            with open(path, 'r') as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [osp.join(root, x.strip()) for x in self.img_files[ds]]
                self.img_files[ds] = list(filter(lambda x: len(x) > 0, self.img_files[ds]))

            self.label_files[ds] = [
                x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                for x in self.img_files[ds]]

        for ds, label_paths in self.label_files.items():
            max_index = -1
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
            self.tid_num[ds] = max_index + 1

        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v

        self.nID = int(last_index + 1)
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.width = img_size[0]
        self.height = img_size[1]
        self.max_objs = opt.K
        self.augment = augment
        self.transforms = transforms

        print('=' * 80)
        print('dataset summary')
        print(self.tid_num)
        print('total # identities:', self.nID)
        print('start index')
        print(self.tid_start_index)
        print('=' * 80)

    # 2023.02.06 同一帧复制2次，不同的数据增强构造正样本
    def __getitem__(self, files_index):
        # box_aug = 0 # 是否进行box增强的标志位
        ret1 = self.getitem(files_index, box_aug=0) # 当前帧 image
        # box_aug = 1
        ret2 = self.getitem(files_index, box_aug=1) # image_pre 2023.03.22 只有这里需要增强box

        return [ret2, ret1]


    # def __getitem__(self, files_index):
    def getitem(self, files_index, box_aug):

        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]

        imgs, labels, img_path, (input_h, input_w) = self.get_data(img_path, label_path, box_aug) # 这一步，目标标签的顺序可能会发生改变
        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                labels[i, 1] += self.tid_start_index[ds]

        output_h = imgs.shape[1] // self.opt.down_ratio # 下采样
        output_w = imgs.shape[2] // self.opt.down_ratio
        num_classes = self.num_classes
        num_objs = labels.shape[0]
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        if self.opt.ltrb:
            wh = np.zeros((self.max_objs, 4), dtype=np.float32)
        else:
            wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs, ), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs, ), dtype=np.uint8)
        ids = np.zeros((self.max_objs, ), dtype=np.int64)
        bbox_xys = np.zeros((self.max_objs, 4), dtype=np.float32)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian
        for k in range(min(num_objs, self.max_objs)):
            label = labels[k]
            bbox = label[2:6] # 是归一化后的值，是相对值 # 2023.02.06 label的第3-6位是Bbox
            cls_id = int(label[0])
            bbox[[0, 2]] = bbox[[0, 2]] * output_w # 乘以下采样后对应的宽，所以得到的是相对于下采样后特征图的值
            bbox[[1, 3]] = bbox[[1, 3]] * output_h
            bbox_amodal = copy.deepcopy(bbox) # box[cx,cy,w,h]
            bbox_amodal[0] = bbox_amodal[0] - bbox_amodal[2] / 2. # x1
            bbox_amodal[1] = bbox_amodal[1] - bbox_amodal[3] / 2. # y1
            bbox_amodal[2] = bbox_amodal[0] + bbox_amodal[2] # x2
            bbox_amodal[3] = bbox_amodal[1] + bbox_amodal[3] # y2
            bbox[0] = np.clip(bbox[0], 0, output_w - 1) # 把bbox[0]限定在最小值0和最大值output_w-1之间
            bbox[1] = np.clip(bbox[1], 0, output_h - 1)
            h = bbox[3]
            w = bbox[2]

            bbox_xy = copy.deepcopy(bbox)  # cxcy ——> xyxy
            bbox_xy[0] = bbox_xy[0] - bbox_xy[2] / 2
            bbox_xy[1] = bbox_xy[1] - bbox_xy[3] / 2
            bbox_xy[2] = bbox_xy[0] + bbox_xy[2]
            bbox_xy[3] = bbox_xy[1] + bbox_xy[3]

            # if h > 0 and w > 0:
            tag = int(label[-1]) # 2023.02.06 有效框的标记位
            if tag > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = 6 if self.opt.mse_loss else radius
                #radius = max(1, int(radius)) if self.opt.mse_loss else radius
                ct = np.array(
                    [bbox[0], bbox[1]], dtype=np.float32) # OpenCV读取的图像矩阵坐标与图像坐标系下的坐标横纵正好相反
                ct_int = ct.astype(np.int32) # （x，y），同时是[列号，行号], 向下取整
                draw_gaussian(hm[cls_id], ct_int, radius)
                if self.opt.ltrb:
                    wh[k] = ct[0] - bbox_amodal[0], ct[1] - bbox_amodal[1], \
                            bbox_amodal[2] - ct[0], bbox_amodal[3] - ct[1]
                else:
                    wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0] # 行号*下采样后的宽 + 列号
                reg[k] = ct - ct_int # 真正的中心点相对于坐标的偏移
                reg_mask[k] = 1
                ids[k] = label[1]
                bbox_xys[k] = bbox_xy

        ret = {'input': imgs, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg, 'ids': ids, 'bbox': bbox_xys}
        return ret


class DetDataset(LoadImagesAndLabels):  # for training
    def __init__(self, root, paths, img_size=(1088, 608), augment=False, transforms=None):

        dataset_names = paths.keys()
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        for ds, path in paths.items():
            with open(path, 'r') as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [osp.join(root, x.strip()) for x in self.img_files[ds]]
                self.img_files[ds] = list(filter(lambda x: len(x) > 0, self.img_files[ds]))

            self.label_files[ds] = [
                x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                for x in self.img_files[ds]]

        for ds, label_paths in self.label_files.items():
            max_index = -1
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
            self.tid_num[ds] = max_index + 1

        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v

        self.nID = int(last_index + 1)
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms

        print('=' * 80)
        print('dataset summary')
        print(self.tid_num)
        print('total # identities:', self.nID)
        print('start index')
        print(self.tid_start_index)
        print('=' * 80)

    def __getitem__(self, files_index):

        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

        imgs, labels, img_path, (h, w) = self.get_data(img_path, label_path)
        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                labels[i, 1] += self.tid_start_index[ds]

        return imgs, labels0, img_path, (h, w)


