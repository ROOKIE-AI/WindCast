# 风力发电预测系统

## 项目简介
本项目基于龙源电力集团提供的SDWPF（Sustainable Development Wind Power Forecasting）数据集，构建了一个风力发电预测系统。该系统利用机器学习方法，通过分析风速、温度等多个特征，对风力涡轮机的发电量进行预测。

## 主要功能
1. 数据预处理与清洗
   - 异常值检测与处理
   - 缺失值处理
   - 特征工程
2. 特征工程
   - 角度特征转换（正弦/余弦编码）
   - 时间特征编码
   - 温度数据修正
   - 冗余特征删除
3. 预测模型
   - 线性回归模型（单变量/多变量）
   - 神经网络模型
   - 特征重要性分析

## 项目结构

## 注意事项
- 数据集包含134个风力涡轮机的信息
- 默认选择发电量最高的前10个涡轮机进行分析
- 部分特征（如温度）可能存在异常值，已在预处理阶段处理

## 技术栈
- Python 3.8+
- 核心库：
  - pandas: 数据处理
  - numpy: 数值计算
  - matplotlib & seaborn: 数据可视化
  - scikit-learn: 机器学习模型
  - PyTorch: 深度学习模型
  - SHAP: 模型解释性分析

## 特征说明
- Wspd: 风速
- Wdir: 风向
- Etmp: 环境温度
- Itmp: 内部温度
- Ndir: 机舱方向
- Pab: 叶片角度
- Prtv: 相对功率
- Patv: 有功功率（预测目标）

## 模型评估
- 使用平均绝对误差(MAE)作为评估指标
- 提供预测值与实际值的可视化对比
- 包含特征重要性分析

## 交互式功能
- 支持选择不同的风力涡轮机进行分析
- 可自定义选择预测特征
- 实时可视化模型预测结果

## 使用说明
1. 克隆项目到本地
2. 安装所需依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 打开 `Wind_Energy_Design_1.ipynb` 开始实验
4. 运行 Jupyter Notebook：
   ```bash
   jupyter notebook
   ```

## 未来改进方向
1. 引入更多高级机器学习模型
2. 增加时序预测功能
3. 优化特征工程方法
4. 提供模型部署方案

## 参考资料
- [SDWPF数据集论文](https://arxiv.org/abs/2208.04360)
- [时间特征编码方法](https://developer.nvidia.com/blog/three-approaches-to-encoding-time-information-as-features-for-ml-models/)
