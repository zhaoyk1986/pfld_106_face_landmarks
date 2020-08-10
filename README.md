# pfld_106_face_landmarks
106点人脸关键点检测的PFLD算法实现

- 数据集准备

  ```bash
  # 下载数据集到data/imgs下
  cd data
  python prepare.py
  ```

-  训练

  ```bash
  CUDA_VISIBLE_DEVICES=0 python train.py
  ```

  