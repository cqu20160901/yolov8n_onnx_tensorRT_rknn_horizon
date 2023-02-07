# yolov8n_onnx_tensorRT_rknn_horizon

yolov8n 部署版本，后处理用python语言和C++语言形式进行改写，便于移植不同平台（onnx、tensorRT、rknn、Horizon）。

# 文件夹结构说明

yolov8n_onnx：onnx模型、测试图像、测试结果、测试demo脚本

yolov8n_TensorRT：TensorRT版本模型、测试图像、测试结果、测试demo脚本、onnx模型、onnx2tensorRT脚本(tensorRT-7.2.3.4)

yolov8n_rknn：rknn模型、测试（量化）图像、测试结果、onnx2rknn转换测试脚本

yolov8n_horizon：地平线模型、测试（量化）图像、测试结果、转换测试脚本、测试量化后onnx模型脚本

# 测试结果
![image](https://github.com/cqu20160901/yolov8n_onnx_tensorRT_rknn_horizon/blob/main/yolov8_onnx/test_onnx_result.jpg)

说明：推理测试预处理没有考虑等比率缩放，激活函数 SiLU 用 Relu 进行了替换。由于使用的是coco128数据进行训练的，且迭代的次数不多，效果并不是很好，仅供测试流程用。

