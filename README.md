# 手势识别项目 (Hand Gesture Recognition)

本项目是一个基于深度学习的手势识别系统，旨在通过摄像头采集手部图像并识别用户的手势。它适用于人机交互、辅助设备控制等场景。项目支持多种手势分类，具有较高的准确率和良好的实时性能。

## 项目特色

- 支持多种常见手势（如“点赞”、“OK”、“胜利”等）
- 实时识别，响应迅速
- 易于扩展新的手势类别
- 代码结构清晰，易于部署和二次开发

## 环境依赖

- Python >= 3.7
- OpenCV
- TensorFlow 或 PyTorch（请根据您的模型选择相应框架）
- Numpy

安装依赖：

```bash
pip install -r requirements.txt
```

## 文件结构

```
hand-gesture-recognition/
├── data/                # 数据集与样本图片
├── models/              # 已训练好的模型文件
├── src/                 # 核心代码
│   ├── detector.py      # 手部检测模块
│   ├── recognizer.py    # 手势识别模块
│   └── utils.py         # 工具函数
├── main.py              # 主程序入口
├── requirements.txt     # 依赖列表
└── README.md            # 项目说明
```

## 快速开始

1. **准备模型与数据集**

   - 请将训练好的模型文件放入 `models/` 目录。
   - 可以使用 `data/` 目录中的样本图片进行测试，或自定义数据集进行训练。

2. **运行主程序**

   ```bash
   python main.py
   ```

   程序会自动调用摄像头进行实时手势识别。

3. **命令行参数说明**

   - `--model_path`: 指定模型文件路径（默认：`models/model.h5`）
   - `--camera_id`: 指定摄像头编号（默认：0）

   示例：

   ```bash
   python main.py --model_path models/model.h5 --camera_id 0
   ```

## 训练自己的模型

如需训练自己的手势识别模型，请参考 `src/` 目录下的相关代码，并准备包含各类手势的图片数据集。建议使用 TensorFlow 或 PyTorch 框架，训练完成后将模型保存至 `models/` 目录。

## 常见问题

- **摄像头无法打开？**  
  请确认摄像头已连接并且驱动正常，尝试更换 `--camera_id` 参数。

- **识别结果不准确？**  
  请确保光线充足，手势完整露出。可以通过增加训练数据或微调模型提升准确率。

## 贡献方式

欢迎大家提交PR或提出Issue完善本项目。建议包括：

- 新的手势类别
- 优化识别算法
- 增加更多平台的支持

## 许可证

本项目采用 MIT License。

---

如有疑问请联系项目作者：[Lllkkll0](https://github.com/Lllkkll0)
