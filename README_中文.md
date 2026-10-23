# CSA 代码说明

对应论文：Multiclass Circle of Willis Segmentation Enhanced by Connectivity-Aware Loss and Attention Mechanisms。

本仓库仅发布代码、开发集划分和运行配置，不包含模型权重、影像、标签、训练日志或实验结果文件。完整安装与运行流程见 [英文说明](README.md)。

- `nnunet_overlay/`：CAL、RSA、ECA、骨架数据加载及八组 2000 轮训练器及正文 CAL 参数对照，仅保留正式配置所需实现及依赖类。
- `configs/`：125 名配对患者的五折划分、ROI 网络计划和标签定义。
- `scripts/install_overlay.py`：将扩展装入独立的 nnU-Net 2.8.1 环境。
- `scripts/prepare_data.py`：定位器数据和按折预测 ROI 数据准备。
- `scripts/train.py`：单折训练入口，支持 CSA 及消融组。
- `scripts/prepare_inference_roi.py`：仅用定位器预测裁剪待推理影像，无需参考标签。
- `scripts/ensemble_probabilities.py`：恢复各折 ROI 概率至原图后集成。
- `scripts/postprocess.py`：PCA/PCom 固定后处理。
- `scripts/evaluate.py`：Dice、clDice、HD95、ASSD 评价。
- `tests/`：CPU 前向、损失梯度、指标边界条件与几何恢复测试。

公开训练入口使用 `Clean...2000` 类；内部部分基类名称含 `1000`，正式入口会将训练预算统一设为 2000 轮。探索性采样变体及未采用方案不在本仓库中。

代码发布不代表已经重新完成全套训练或独立复现论文数值。本轮只进行源码与合成输入测试，不改动服务器上的原训练项目。测试通过也不能替代真实数据上的端到端验证。

数据从 TopCoW 官方页面获取，并遵循各子集的许可和引用要求。权重不随本仓库提供；推理前需要自行训练或取得兼容检查点。
