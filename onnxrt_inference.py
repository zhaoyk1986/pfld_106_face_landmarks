#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/8/8 2:06 下午
# @Author : Xintao
# @File : onnxrt_inference.py
import numpy as np
import onnxruntime
import time
import cv2


onnx_model_path = "./output/v3.onnx"
img_path = "./1.png"
img = cv2.imread(img_path)
show_img = True

# 网络输入是BGR格式的图片
img1 = cv2.resize(img, (112, 112))
image_data = img1.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255


session = onnxruntime.InferenceSession(onnx_model_path, None)
# get the name of the first input of the model
input_name = session.get_inputs()[0].name
import time
tic = time.time()
for i in range(100):
    output = session.run([], {input_name: image_data})[1]

t = (time.time() - tic) / 100
print('average infer time: {:.4f}ms, FPS: {:.2f}'.format(t * 1000, 1 / t))
print('output.shape: ', output.shape)
# print(output[0])
if show_img:
    landmarks = output.reshape(-1, 2)
    landmarks[:, 0] = landmarks[:, 0] * img.shape[1]
    landmarks[:, 1] = landmarks[:, 1] * img.shape[0]
    img_copy = img.copy().astype(np.uint8)
    for (x, y) in landmarks:
        cv2.circle(img_copy, (int(x), int(y)), 2, (0, 0, 255), -1)
    cv2.imshow('demo', img_copy)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # cv2.imwrite('result1.jpg', img_copy)
