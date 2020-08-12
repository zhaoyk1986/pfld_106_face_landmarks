# pfld_106_face_landmarks
106点人脸关键点检测的PFLD算法实现

- [ ] 预训练权重
- [ ] 性能测试 
- [x] update GhostNet
- [x] update MobileNetV3 

- 数据集准备

  ```bash
  # 下载数据集到data/imgs下
  cd data
  python prepare.py
  ```
  ```bash
  # data 文件夹结构
  data/
    imgs/
    train_data/
      imgs/
      list.txt
    test_data/
      imgs/
      list.txt
  ```
  
-  训练

  ```bash
  CUDA_VISIBLE_DEVICES=0 python train.py
  ```

  