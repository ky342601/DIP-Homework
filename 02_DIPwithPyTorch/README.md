# Assignment 2 - DIP with PyTorch

---
## Training
### 1. 使用 PyTorch 实现泊松图像编辑
填充'run_blending_gradio.py'中的[多边形生成掩码](run_blending_gradio.py#L95)与[Laplacian距离计算](run_blending_gradio.py#L115)部分，然后运行：

```run
python run_blending_gradio.py
```

### 2. Pix2Pix implementation.
实现带有完全卷积图层的Pix2Pix，填充 [全卷积神经网络](FCN_network.py#L3)部分，然后在`./Pix2Pix`下运行：

```bash
bash download_facades_dataset.sh
python train.py
```

---
## Requirements
安装要求：
```setup
pip install -r requirements.txt
```

---
## Results

以下为使用facades_dataset的部分训练结果：

<img src="Pix2Pix\train_results\epoch_295\result_1.png" alt="alt text" width="800">
<img src="Pix2Pix\train_results\epoch_295\result_2.png" alt="alt text" width="800">
<img src="Pix2Pix\train_results\epoch_295\result_3.png" alt="alt text" width="800">
<img src="Pix2Pix\train_results\epoch_295\result_4.png" alt="alt text" width="800">
<img src="Pix2Pix\train_results\epoch_295\result_5.png" alt="alt text" width="800">

300次迭代后，验证集平均损失降至0.3379左右。