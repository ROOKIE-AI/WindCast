import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import metrics
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from ipywidgets import interact
from typing import Callable, List, Tuple, Dict, Optional
import pickle

# 定义图表中各类文本的字体大小常量，用于保持绘图风格的一致性
FONT_SIZE_TICKS = 15       # 坐标轴刻度标签字体大小
FONT_SIZE_TITLE = 25       # 图表标题字体大小
FONT_SIZE_AXES = 20        # 坐标轴标签字体大小

def prepare_data(df: pd.core.frame.DataFrame, turb_id: int) -> pd.core.frame.DataFrame:
    """
    在将数据输入神经网络之前对其进行预处理
    
    参数:
        df (pandas.core.frame.DataFrame): 来自上一个实验的数据。
        turb_id (int): 要使用的风力涡轮机的ID。
        
    返回:
        pandas.core.frame.DataFrame: 预处理后的数据
    """
    # 按照每6条数据采样1条数据（对应每小时数据）
    df = df[5::6]
    # 筛选指定涡轮机的行
    df = df[df.TurbID == turb_id]
    # 删除涡轮机ID列，因为后续不再需要
    df = df.drop(["TurbID"], axis=1)
    # 将"Datetime"列转换为datetime格式，并设置为索引，同时删除该列
    df.index = pd.to_datetime(df.pop("Datetime"), format="%Y-%m-%d %H:%M:%S")
    # 对Include列为False的行设置缺失值标记（这里用-1表示异常）
    df = df.mask(df.Include == False, -1)
    # 删除Include列，完成异常值处理
    df = df.drop(["Include"], axis=1)
    # 按照固定顺序重新排列列，保证数据顺序一致
    df = df[
        [
            "Wspd",    # 风速
            "Etmp",    # 环境温度
            "Itmp",    # 内部温度
            "Prtv",    # 有功功率
            "WdirCos", # 风向余弦
            "WdirSin", # 风向正弦
            "NdirCos", # 方向余弦
            "NdirSin", # 方向正弦
            "PabCos",  # 功率余弦
            "PabSin",  # 功率正弦
            "Patv",    # 活动功率（目标变量）
        ]
    ]

    return df

def normalize_data(
    train_data: pd.core.frame.DataFrame,
    val_data: pd.core.frame.DataFrame,
    test_data: pd.core.frame.DataFrame,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, pd.core.series.Series, pd.core.series.Series
]:
    """
    对训练集、验证集和测试集进行标准化

    参数:
        train_data (pd.core.frame.DataFrame): 训练集数据。
        val_data (pd.core.frame.DataFrame): 验证集数据。
        test_data (pd.core.frame.DataFrame): 测试集数据。

    返回:
        tuple: 标准化后的训练、验证、测试数据，以及训练集的均值和标准差
    """
    # 计算训练集均值和标准差，后续用于标准化其他数据集
    train_mean = train_data.mean()
    train_std = train_data.std()

    # 对训练、验证、测试集按同一标准进行归一化处理
    train_data = (train_data - train_mean) / train_std
    val_data = (val_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    return train_data, val_data, test_data, train_mean, train_std

@dataclass
class DataSplits:
    """数据分割封装类, 包含normalize和unnormalize的训练集、验证集和测试集"""
    train_data: pd.core.frame.DataFrame         # 标准化后的训练集数据
    val_data: pd.core.frame.DataFrame           # 标准化后的验证集数据
    test_data: pd.core.frame.DataFrame          # 标准化后的测试集数据
    train_mean: pd.core.series.Series           # 训练集各特征的均值
    train_std: pd.core.series.Series            # 训练集各特征的标准差
    train_df_unnormalized: pd.core.frame.DataFrame  # 原始训练集（未标准化）
    val_df_unnormalized: pd.core.frame.DataFrame    # 原始验证集（未标准化）
    test_df_unnormalized: pd.core.frame.DataFrame   # 原始测试集（未标准化）

def train_val_test_split(df: pd.core.frame.DataFrame) -> DataSplits:
    """将数据框分成训练、验证和测试集。

    参数:
        df (pd.core.frame.DataFrame): 待分割的数据框。

    返回:
        DataSplits: 包含标准化及非标准化数据集的封装对象
    """
    n = len(df)
    # 按70%、20%、10%比例划分训练、验证、测试集
    train_df = df[0 : int(n * 0.7)]
    val_df = df[int(n * 0.7) : int(n * 0.9)]
    test_df = df[int(n * 0.9) :]

    # 对每个数据集进行深拷贝，防止后续数据修改影响原数据
    train_df_un = train_df.copy(deep=True)
    val_df_un = val_df.copy(deep=True)
    test_df_un = test_df.copy(deep=True)
    
    # 利用mask函数处理异常值，将Patv为-1的值替换为NaN
    train_df_un = train_df_un.mask(train_df_un.Patv == -1, np.nan)
    val_df_un = val_df_un.mask(val_df_un.Patv == -1, np.nan)
    test_df_un = test_df_un.mask(test_df_un.Patv == -1, np.nan)

    # 使用normalize_data对数据集进行标准化处理
    train_df, val_df, test_df, train_mn, train_st = normalize_data(
        train_df, val_df, test_df
    )

    # 构建DataSplits对象，封装所有数据集信息
    ds = DataSplits(
        train_data=train_df,
        val_data=val_df,
        test_data=test_df,
        train_mean=train_mn,
        train_std=train_st,
        train_df_unnormalized=train_df_un,
        val_df_unnormalized=val_df_un,
        test_df_unnormalized=test_df_un,
    )

    return ds

def plot_time_series(data_splits: DataSplits) -> None:
    """
    绘制目标特征和预测特征的时间序列

    参数:
        data_splits (DataSplits): 包含未标准化数据的对象，用于绘图以显示真实数据尺度
    """
    # 解包未归一化的训练、验证、测试数据，用于展示原始数值
    train_df, val_df, test_df = (
        data_splits.train_df_unnormalized,
        data_splits.val_df_unnormalized,
        data_splits.test_df_unnormalized,
    )

    def plot_time_series(feature):
        # 创建包含两个子图的绘图区域：上图显示目标变量，下图显示预测变量
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
        
        # 绘制目标特征 'Patv' 的时间序列（蓝色、绿色、红色分别代表训练、验证、测试集）
        ax1.plot(train_df["Patv"], color="blue", label="training")
        ax1.plot(val_df["Patv"], color="green", label="validation")
        ax1.plot(test_df["Patv"], color="red", label="test")
        ax1.set_title("Time series of Patv (target)", fontsize=FONT_SIZE_TITLE)
        ax1.set_ylabel("Active Power (kW)", fontsize=FONT_SIZE_AXES)
        ax1.set_xlabel("Date", fontsize=FONT_SIZE_AXES)
        ax1.legend(fontsize=15)
        ax1.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)

        # 绘制预测特征的时间序列
        ax2.plot(train_df[feature], color="blue", label="training")
        ax2.plot(val_df[feature], color="green", label="validation")
        ax2.plot(test_df[feature], color="red", label="test")
        ax2.set_title(f"Time series of {feature} (predictor)", fontsize=FONT_SIZE_TITLE)
        ax2.set_ylabel(f"{feature}", fontsize=FONT_SIZE_AXES)
        ax2.set_xlabel("Date", fontsize=FONT_SIZE_AXES)
        ax2.legend(fontsize=15)
        ax2.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)

        # 自动调整各子图间距，防止重叠
        plt.tight_layout()
        plt.show()

    # 创建下拉菜单供用户选择预测特征（排除目标"Patv"）
    feature_selection = widgets.Dropdown(
        options=[f for f in list(train_df.columns) if f != "Patv"],
        description="Feature",
    )

    # 利用交互组件，使得选择菜单与绘图函数联动
    interact(plot_time_series, feature=feature_selection)

def compute_metrics(
    true_series: np.ndarray, forecast: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """计算两个时间序列的均方误差和平均绝对误差。

    参数：
        true_series (np.ndarray): 真实值数组。
        forecast (np.ndarray): 预测值数组。
        
    返回：
        tuple: 均方误差（MSE）和平均绝对误差（MAE）
    """
    # 利用TensorFlow定义均方误差指标
    mse_metric = tf.keras.metrics.MeanSquaredError()
    # 利用TensorFlow定义平均绝对误差指标
    mae_metric = tf.keras.metrics.MeanAbsoluteError()

    mse = mse_metric(true_series, forecast).numpy()
    mae = mae_metric(true_series, forecast).numpy()

    return mse, mae

class WindowGenerator:
    """用于处理时间序列窗口生成器，包括生成训练、验证、测试集窗口及其可视化"""

    def __init__(
        self,
        input_width,
        label_width,
        shift,
        train_df,
        val_df,
        test_df,
        label_columns=["Patv"],
    ):
        """
        初始化窗口生成器
        
        参数:
            input_width: 输入序列的长度（时间步数量）
            label_width: 标签序列的长度（待预测的时间步数量）
            shift: 输入与标签之间的时移量
            train_df: 训练集数据（DataFrame格式）
            val_df: 验证集数据
            test_df: 测试集数据
            label_columns: 指定作为预测目标的列（默认["Patv"]）
        """
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns
        if label_columns is not None:
            # 建立标签列名到索引的映射字典
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }
        # 建立所有特征列到索引的映射字典
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        # 总窗口长度 = 输入长度 + 偏移量（预测步数）
        self.total_window_size = input_width + shift

        # 定义输入数据在窗口内的切片
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        # 定义标签数据在窗口内的起始位置及切片
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        """
        将一个完整的窗口数据分割为输入和标签两部分
        参数:
            features: 形状为[样本数, 窗口总长度, 特征数]的张量
        返回:
            inputs: 输入部分数据
            labels: 标签部分数据（也可仅保留指定标签列）
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            # 根据预设的标签列顺序，重排标签数据
            labels = tf.stack(
                [
                    labels[:, :, self.column_indices[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )

        # 显式设置输入和标签数据的形状
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def plot(self, model=None, plot_col="Patv", max_subplots=1):
        """
        可视化窗口中的输入和标签数据；如果传入模型，则同时展示预测结果。
        
        参数:
            model: 可选模型，用于生成预测结果
            plot_col: 待绘制的特征列名称
            max_subplots: 至多显示的子图数
        """
        inputs, labels = self.example
        plt.figure(figsize=(20, 6))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.title("Inputs (past) vs Labels (future predictions)", fontsize=FONT_SIZE_TITLE)
            plt.ylabel(f"{plot_col} (normalized)", fontsize=FONT_SIZE_AXES)
            plt.plot(
                self.input_indices,
                inputs[n, :, plot_col_index],
                color="green",
                linestyle="--",
                label="Inputs",
                marker="o",
                markersize=10,
                zorder=-10,
            )

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.plot(
                self.label_indices,
                labels[n, :, label_col_index],
                color="orange",
                linestyle="--",
                label="Labels",
                markersize=10,
                marker="o"
            )
            if model is not None:
                predictions = model(inputs)
                plt.scatter(
                    self.label_indices,
                    predictions[n, :, label_col_index],
                    marker="*",
                    edgecolors="k",
                    label="Predictions",
                    c="pink",
                    s=64,
                )
            plt.legend(fontsize=FONT_SIZE_TICKS)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        plt.xlabel("Timestep", fontsize=FONT_SIZE_AXES)

    def plot_long(
        self,
        model,
        data_splits,
        plot_col="Patv",
        time_steps_future=1,
        baseline_mae=None,
    ):
        # 从数据拆分对象中提取训练集的均值和标准差，用于反归一化处理
        train_mean, train_std = data_splits.train_mean, data_splits.train_std
        self.test_size = len(self.test_df)
        # 生成测试集数据（不打乱顺序）
        self.test_data = self.make_test_dataset(self.test_df, self.test_size)

        inputs, labels = next(iter(self.test_data))

        plt.figure(figsize=(20, 6))
        plot_col_index = self.column_indices[plot_col]

        plt.ylabel(f"{plot_col} (kW)", fontsize=FONT_SIZE_AXES)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        # 将标签数据反标准化，还原为原始单位
        labels = (labels * train_std.Patv) + train_mean.Patv

        # 根据预测小时数确定反归一化后数据的绘制区间
        upper = 24 - (time_steps_future - 1)
        lower = self.label_indices[-1] - upper
        self.label_indices_long = self.test_df.index[lower:-upper]

        plt.plot(
            self.label_indices_long[:],
            labels[:, time_steps_future - 1, label_col_index][:],
            label="Labels",
            c="green",
        )

        if model is not None:
            predictions = model(inputs)
            predictions = (predictions * train_std.Patv) + train_mean.Patv
            predictions_for_timestep = predictions[
                :, time_steps_future - 1, label_col_index
            ][:]
            predictions_for_timestep = tf.nn.relu(predictions_for_timestep).numpy()
            plt.plot(
                self.label_indices_long[:],
                predictions_for_timestep,
                label="Predictions",
                c="orange",
                linewidth=3,
            )
            plt.legend(fontsize=FONT_SIZE_TICKS)
            _, mae = compute_metrics(
                labels[:, time_steps_future - 1, label_col_index][:],
                predictions_for_timestep,
            )

            if baseline_mae is None:
                baseline_mae = mae

            print(
                f"\nMean Absolute Error (kW): {mae:.2f} for forecast.\n\nImprovement over random baseline: {100*((baseline_mae -mae)/baseline_mae):.2f}%"
            )
        plt.title("Predictions vs Real Values for Test Split", fontsize=FONT_SIZE_TITLE)
        plt.xlabel("Date", fontsize=FONT_SIZE_AXES)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        
        return mae

    def make_test_dataset(self, data, bs):
        # 将DataFrame转换为float32类型的ndarray
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=bs,
        )

        ds = ds.map(self.split_window)

        return ds

    def make_dataset(self, data):
        # 将输入数据转换为numpy数组，并确保其数据类型为float32，这有助于后续的数值计算
        data = np.array(data, dtype=np.float32)
        """
        使用TensorFlow的timeseries_dataset_from_array函数生成时间序列数据集
        参数说明：
          data: 已转换为numpy数组的原始数据
          targets: 设置为None，因为本例中数据集不包含单独的目标标签
          sequence_length: 每个时间窗口的长度，由self.total_window_size属性定义，
                           包括输入序列和预测步长
          sequence_stride: 窗口间移动的步长，这里设为1，表示窗口按时间步逐个滑动
          shuffle: 设为True，使得数据在生成过程中被随机打乱，增强模型训练效果
          batch_size: 每个批次包含32个样本
        """
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,
        )
        
        # 对生成的数据集应用映射函数split_window
        # 该函数用于将每个完整的时间窗口划分为输入部分和对应的标签部分
        ds = ds.map(self.split_window)
        
        # 返回已处理完成的时间序列数据集，供模型训练或预测使用
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        result = getattr(self, "_example", None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result

def generate_window(  
    train_df: pd.core.frame.DataFrame,  
    val_df: pd.core.frame.DataFrame,  
    test_df: pd.core.frame.DataFrame,  
    days_in_past: int,  
    width: int = 24  
) -> WindowGenerator:  
    """  
    根据训练集、验证集、测试集划分以及过去的天数，创建一个时间窗口化的数据集。  

    参数:  
        train_df (pd.core.frame.DataFrame): 训练集。  
        val_df (pd.core.frame.DataFrame): 验证集。  
        test_df (pd.core.frame.DataFrame): 测试集。  
        days_in_past (int): 用于预测的过去天数，即看过去多少天的特征来预测未来。  
        width (int, optional): 每天的时间步宽度，默认为 24（一天24小时）。  

    返回:  
        WindowGenerator: 配置好的时间窗口生成器对象。  
    """  
    # 定义预测步长为24小时
    OUT_STEPS = 24  
    
    # 生成WindowGenerator对象，输入宽度为过去天数乘以每天的小时数，标签宽度为固定的24小时，shift设置为24
    multi_window = WindowGenerator(  
        input_width=width * days_in_past,  
        label_width=OUT_STEPS,             
        train_df=train_df,                 
        val_df=val_df,                     
        test_df=test_df,                   
        shift=OUT_STEPS,                   
    )  
    
    return multi_window

def create_model(num_features: int, days_in_past: int, ) -> tf.keras.Model:
    """创建一个用于时间序列预测的 Conv-LSTM 模型

    参数:
        num_features (int): 输入特征数量
        days_in_past (int): 使用过去多少天的数据来预测未来
        
    返回:
        tf.keras.Model: 构建但未编译的模型
    """
    CONV_WIDTH = 3   # 卷积层卷积核宽度
    OUT_STEPS = 24   # 输出步长（预测未来24小时）
    model = tf.keras.Sequential(
        [
            # 掩码层，用于忽略值为-1.0的输入（异常数据）
            tf.keras.layers.Masking(
                mask_value=-1.0, input_shape=(days_in_past * 24, num_features)
            ),
            # Lambda层选取窗口中最后CONV_WIDTH个时间步作为卷积输入
            # tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
            # Conv1D层提取局部时序特征
            tf.keras.layers.Conv1D(256, activation="relu", kernel_size=(CONV_WIDTH)),
            # 第一层双向LSTM，用于捕捉时序依赖性，返回所有时间步的输出
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(32, return_sequences=True)
            ),
            # 第二层双向LSTM，仅返回最终输出，进一步提取序列信息
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(32, return_sequences=False)
            ),
            # 全连接层，将LSTM输出映射到预测目标维度
            tf.keras.layers.Dense(
                OUT_STEPS * 1, kernel_initializer=tf.initializers.zeros()
            ),
            # 重塑输出为[OUT_STEPS, 1]的形状
            tf.keras.layers.Reshape([OUT_STEPS, 1]),
        ]
    )

    return model

def compile_and_fit(
    model: tf.keras.Model, window: WindowGenerator, patience: int = 2
) -> tf.keras.callbacks.History:
    """
    编译并训练模型，利用早停机制防止过拟合。

    参数:
        model (tf.keras.Model): 待训练模型。
        window (WindowGenerator): 包含窗口化训练数据的生成器。
        patience (int, optional): 早停轮数，若验证损失连续patience轮未改善则停止训练。

    返回:
        tf.keras.callbacks.History: 训练历史记录。
    """
    EPOCHS = 20   # 最大训练轮数
    
    # 定义EarlyStopping回调监控验证集损失
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min"
    )

    # 编译模型，采用均方误差作为损失函数和Adam优化器
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
    )

    # 固定随机种子以保证实验结果的重复性
    tf.random.set_seed(432)
    np.random.seed(432)
    random.seed(432)

    # 训练模型，传入训练数据、验证数据以及早停回调
    history = model.fit(
        window.train, epochs=EPOCHS, validation_data=window.val, callbacks=[early_stopping]
    )
    
    # 如果训练轮数少于设定的最大轮数，则打印提示信息
    if len(history.epoch) < EPOCHS:
        print("\nTraining stopped early to prevent overfitting, as the validation loss is increasing for two consecutive steps.")
    
    return history

def train_conv_lstm_model(
    data: pd.core.frame.DataFrame, features: List[str], days_in_past: int
) -> Tuple[WindowGenerator, tf.keras.Model, DataSplits]:
    """训练用于时间序列预测的Conv-LSTM模型

    参数:
        data (pd.core.frame.DataFrame): 包含所有数据的数据框
        features (list[str]): 用于预测的特征列表
        days_in_past (int): 使用过去多少天数据来预测未来24小时数据
        
    返回:
        tuple: 包括窗口生成器、预测模型和数据拆分对象
    """
    # 根据指定特征拆分数据集，同时进行标准化与保留原始数据
    data_splits = train_val_test_split(data[features])

    train_data, val_data, test_data, train_mean, train_std = (
        data_splits.train_data,
        data_splits.val_data,
        data_splits.test_data,
        data_splits.train_mean,
        data_splits.train_std,
    )

    # 生成时间窗口数据生成器
    window = generate_window(train_data, val_data, test_data, days_in_past)
    
    num_features = window.train_df.shape[1]

    # 构建Conv-LSTM模型
    model = create_model(num_features, days_in_past)
    # 编译并训练模型
    history = compile_and_fit(model, window)
    
    return window, model, data_splits



def prediction_plot(
    func: Callable, model: tf.keras.Model, data_splits: DataSplits, baseline_mae: float
) -> None:
    """绘制交互式预测与真实值对比图。

    参数:
        func (Callable): 应封装WindowGenerator的plot_long方法的闭包函数。
        model (tf.keras.Model): 已训练的模型。
        data_splits (DataSplits): 包含数据拆分信息的对象。
        baseline_mae (float): 基线模型的平均绝对误差，用于对比。
    """
    def _plot(time_steps_future):
        """根据用户指定的预测未来小时数绘图。
        
        参数:
            time_steps_future: 预测未来的数据小时数
        """
        mae = func(
            model,
            data_splits,
            time_steps_future=time_steps_future,
            baseline_mae=baseline_mae,
        )

    # 创建交互式滑条用于选择预测未来的小时数
    time_steps_future_selection = widgets.IntSlider(
        value=24,
        min=1,
        max=24,
        step=1,
        description="Hours into future",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        layout={"width": "500px"},
        style={"description_width": "initial"},
    )

    interact(_plot, time_steps_future=time_steps_future_selection)
    
def random_forecast(  
    data_splits: DataSplits, n_days: int = 1  
) -> Tuple[WindowGenerator, tf.keras.Model]:
    """  
    为时间窗口生成随机预测。  

    参数:  
        data_splits (DataSplits): 数据拆分对象，包含训练、验证和测试集。  
        n_days (int, optional): 用于随机预测的历史天数，默认为1天。  

    返回:  
        tuple: 包含生成的时间窗口和随机基线预测模型。  
    """  
    train_data, val_data, test_data = (  
        data_splits.train_data,  
        data_splits.val_data,  
        data_splits.test_data,  
    )  

    # 使用指定天数生成时间窗口
    random_window = generate_window(train_data, val_data, test_data, n_days)  

    # 定义随机基线模型，利用随机打乱输入数据生成预测
    class randomBaseline(tf.keras.Model):
        def call(self, inputs):
            tf.random.set_seed(424)
            np.random.seed(424)
            random.seed(424)
            stacked = tf.random.shuffle(inputs)
            return stacked[:, :, -1:]  

    random_baseline = randomBaseline()
    random_baseline.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    return random_window, random_baseline

def repeat_forecast(
    data_splits: DataSplits, shift: int = 24
) -> Tuple[WindowGenerator, tf.keras.Model]:
    """
    执行重复预测逻辑。

    参数:
        data_splits (DataSplits): 数据拆分对象。
        shift (int): 窗口偏移量，默认24小时。
        
    返回:
        tuple: 包含窗口生成器和简单重复预测模型
    """
    train_data, val_data, test_data = (
        data_splits.train_data,
        data_splits.val_data,
        data_splits.test_data,
    )
    repeat_window = generate_window(train_data, val_data, test_data, 1, shift)

    class RepeatBaseline(tf.keras.Model):
        def call(self, inputs):
            return inputs[:, :, -1:]

    repeat_baseline = RepeatBaseline()
    repeat_baseline.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    return repeat_window, repeat_baseline

def interact_repeat_forecast(
    data_splits: DataSplits, baseline_mae: float
) -> None:
    """
    绘制预测与真实值的交互式可视化图表。

    参数:
        data_splits (DataSplits): 数据拆分对象。
        baseline_mae (float): 基准MAE值，用于评价模型预测效果。
    """
    def _plot(shift):
        repeat_window, repeat_baseline = repeat_forecast(data_splits, shift=shift)
        _ = repeat_window.plot_long(repeat_baseline, data_splits, baseline_mae=baseline_mae)

    shift_selection = widgets.IntSlider(
        value=24,
        min=1,
        max=24,
        step=1,
        description="Hours into future",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        layout={"width": "500px"},
        style={"description_width": "initial"},
    )

    interact(_plot, shift=shift_selection)

def moving_avg_forecast(data_splits: DataSplits, n_days: int) -> Tuple[WindowGenerator, tf.keras.Model]:
    """
    执行移动平均预测逻辑。

    参数:
        data_splits (DataSplits): 数据拆分对象。
        n_days (int): 用于计算移动平均的历史天数。
        
    返回:
        tuple: 包含窗口生成器和移动平均预测模型
    """
    train_data, val_data, test_data = (
        data_splits.train_data,
        data_splits.val_data,
        data_splits.test_data,
    )
    moving_avg_window = generate_window(train_data, val_data, test_data, n_days)

    class avgBaseline(tf.keras.Model):
        def call(self, inputs):
            m = tf.math.reduce_mean(inputs, axis=1)
            stacked = tf.stack([m for _ in range(inputs.shape[1])], axis=1)
            return stacked[:, :, -1:]

    moving_avg_baseline = avgBaseline()
    moving_avg_baseline.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    return moving_avg_window, moving_avg_baseline

def add_wind_speed_forecasts(
    df: pd.core.frame.DataFrame, add_noise=False
) -> pd.core.frame.DataFrame:
    """
    生成合成的风速预测。未来的时间越长，这些预测的噪声就越大

    参数:
        df (pd.core.frame.DataFrame): 原始风机数据的数据框
        add_noise (可选): 噪声添加选项，可设为 "linearly_increasing" 或 "mimic_real_forecast"，默认为False（无噪声）
        
    返回:
        pd.core.frame.DataFrame: 包含原始数据及生成的风速预测数据的新数据框
    """
    df_2 = df.copy(deep=True)
    # 定义预测周期，从1小时到29小时步长
    periods = [*range(1, 30, 1)]
    
    for period in periods:
        if add_noise == "linearly_increasing":
            np.random.seed(8752)
            noise_level = 0.2 * period  # 噪声水平与预测步长线性增长
            noise = np.random.randn(len(df)) * noise_level
        elif add_noise == "mimic_real_forecast":
            np.random.seed(8752)
            noise_level = 2 + 0.05 * period  # 模拟真实预报噪声水平
            noise = np.random.randn(len(df)) * noise_level
        else:
            noise = 0
        
        # 对尾部数据进行padding，使得生成的预测数据长度与原数据匹配
        padding_slice = df_2["Wspd"][-period:].to_numpy()
        values = np.concatenate((df_2["Wspd"][period:].values, padding_slice)) + noise
        df_2[f"fc-{period}h"] = values

    return df_2

def plot_forecast_with_noise(
    data_with_wspd_forecasts: pd.core.frame.DataFrame,
) -> None:
    """
    创建一个交互式图表，显示合成预测如何随未来预测范围的变化而变化。

    参数:
        data_with_wspd_forecasts (pd.core.frame.DataFrame): 包含合成风速预测的数据框。
    """
    def _plot(noise_level):
        fig, ax = plt.subplots(figsize=(20, 6))

        df = data_with_wspd_forecasts
        # 根据噪声级别选择对应的预测列，并调整索引以对齐图像
        synth_data = df[f"fc-{noise_level}h"][
            5241 - noise_level : -noise_level
        ].values
        synth_data = tf.nn.relu(synth_data).numpy()
        real_data = df["Wspd"][5241:].values
        real_data = tf.nn.relu(real_data).numpy()

        mae = metrics.mean_absolute_error(real_data, synth_data)
        print(f"\nMean Absolute Error (m/s): {mae:.2f} for forecast\n")
        ax.plot(df.index[5241:], real_data, label="true values")
        ax.plot(
            df.index[5241:],
            synth_data,
            label="syntethic predictions",
        )

        ax.set_title("Generated wind speed forecasts", fontsize=FONT_SIZE_TITLE)
        ax.set_ylabel("Wind Speed (m/s)", fontsize=FONT_SIZE_AXES)
        ax.set_xlabel("Date", fontsize=FONT_SIZE_AXES)
        ax.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        ax.legend()

    noise_level_selection = widgets.IntSlider(
        value=1,
        min=1,
        max=25,
        step=1,
        description="Noise level in m/s (low to high)",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=False,
        layout={"width": "500px"},
        style={"description_width": "initial"},
    )

    interact(_plot, noise_level=noise_level_selection)

def window_plot(data_splits: DataSplits) -> None:
    """
    创建一个交互式图表，显示数据如何根据用于预测未来24小时的数据天数进行窗口化。

    参数:
        data_splits (DataSplits): 数据拆分对象，包含训练、验证和测试集。
    """
    train_data, val_data, test_data = (
        data_splits.train_data,
        data_splits.val_data,
        data_splits.test_data,
    )

    def _plot(time_steps_past):
        window = generate_window(train_data, val_data, test_data, time_steps_past)
        window.plot()

    time_steps_past_selection = widgets.IntSlider(
        value=1,
        min=1,
        max=14,
        step=1,
        description="Days before",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        layout={"width": "500px"},
        style={"description_width": "initial"},
    )

    interact(_plot, time_steps_past=time_steps_past_selection)
    
def load_weather_forecast() -> Dict[str, Dict[List[datetime], List[float]]]:
    """
    加载风速数据和预报数据，并返回字典形式
    
    通过pickle加载包含三地风速和预报数据的文件并返回字典。
    """
    with open("../data/weather_forecast.pkl", "rb") as f:
        weather_forecasts = pickle.load(f)
    return weather_forecasts

def plot_forecast(weather_forecasts: Dict[str, Dict[List[datetime], List[float]]]) -> None:
    """
    创建一个交互式图表，显示风速数据的实际值与预测值的对比。

    参数:
        weather_forecasts (dict): 包含历史天气和预报数据的字典。
    """
    def _plot(city, time_steps_future):
        format_timestamp = "%Y-%m-%d %H:%M:%S"

        weather_forecast = weather_forecasts[city]
        
        dates_real, winds_real = weather_forecast[0]
        dates_real = [datetime.strptime(i, format_timestamp) for i in dates_real]
        dates_forecast, winds_forecast = weather_forecast[time_steps_future]
        dates_forecast = [datetime.strptime(i, format_timestamp) for i in dates_forecast]

        # 固定绘图的最小和最大日期，确保图表范围一致
        min_date = datetime.strptime("2022-11-16 18:00:00", format_timestamp)
        max_date = datetime.strptime("2023-01-11 15:00:00", format_timestamp)
        
        # 调用prepare_wind_data提取两个数据集中在同一时刻的有效数据，并限制在指定日期范围内
        dates_real, dates_forecast, winds_real, winds_forecast = prepare_wind_data(
            dates_real, dates_forecast, winds_real, winds_forecast, min_date, max_date
        )
        
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.plot(dates_real, winds_real, label="Actual windspeed")
        ax.plot(dates_forecast, winds_forecast, label=f"Forecasted windspeed {time_steps_future} Hours in the Future")
        ax.set_title(f"History of Actual vs Forecasted Windspeed in {city}", fontsize=25)
        ax.set_ylabel("Wind Speed (m/s)", fontsize=20)
        ax.set_xlabel("Date", fontsize=20)
        ax.tick_params(axis="both", labelsize=15)
        ax.legend(fontsize=15)
        
        mae = metrics.mean_absolute_error(winds_real, winds_forecast)
        print(f"\nMean Absolute Error (m/s): {mae:.2f} for forecast\n")
       
    city_selection = widgets.Dropdown(
        options=weather_forecasts.keys(),
        description='City',
    )
    time_steps_future_selection = widgets.IntSlider(
        value=1,
        min=3,
        max=120,
        step=3,
        description="Hours into future",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        layout={"width": "500px"},
        style={"description_width": "initial"},
    )

    interact(_plot, city=city_selection, time_steps_future=time_steps_future_selection)
    
def prepare_wind_data(
    dates0: List[datetime],
    dates1: List[datetime],
    winds0: List[float],
    winds1: List[float],
    min_bound: Optional[str] = None,
    max_bound: Optional[str] = None
) -> Tuple[List[datetime], List[datetime], List[float], List[float]]:
    """
    接受两组风速数据，找到在两个数据集中同时出现且满足时间条件的日期，并限制在指定日期范围内

    参数:
        dates0 (list): 第一组数据的日期列表。
        dates1 (list): 第二组数据的日期列表。
        winds0 (list): 第一组数据对应的风速列表。
        winds1 (list): 第二组数据对应的风速列表。
        min_bound (datetime): 可选，限制数据的最小日期。
        max_bound (datetime): 可选，限制数据的最大日期。
        
    返回:
        tuple: 重叠数据的日期及对应风速列表。
    """
    winds0_overlap = []
    winds1_overlap = []
    dates0_overlap = []
    dates1_overlap = []
    
    # 遍历第一组数据，查找同时在第二组中出现且满足时间条件的日期
    for date, wind in zip(dates0, winds0):
        if (date in dates1 and 
            (min_bound is None or date > min_bound) and
            (max_bound is None or date < max_bound)
           ):
            winds0_overlap.append(wind)
            dates0_overlap.append(date)
    # 遍历第二组数据，进行同样的筛选
    for date, wind in zip(dates1, winds1):
        if (date in dates0 and 
            (min_bound is None or date > min_bound) and
            (max_bound is None or date < max_bound)
           ):
            winds1_overlap.append(wind)
            dates1_overlap.append(date)
    
    return dates0_overlap, dates1_overlap, winds0_overlap, winds1_overlap

def plot_mae_forecast(weather_forecasts: Dict[str, Dict[List[datetime], List[float]]]) -> None:
    """
    创建一个交互式图表，显示风速预测的MAE随预测时间步长的变化

    参数:
        weather_forecasts (dict): 包含历史天气和预报数据的字典。
    """
    def _plot(city):
        weather_forecast = weather_forecasts[city]
        
        times = sorted(weather_forecast.keys())[1::]
        maes = []
        
        dates_real, winds_real = weather_forecast[0]
        for time in times:
            dates_forecast, winds_forecast = weather_forecast[time]
            dates_real, dates_forecast, winds_real, winds_forecast = prepare_wind_data(
                dates_real, dates_forecast, winds_real, winds_forecast
            )
            mae = metrics.mean_absolute_error(winds_real, winds_forecast)
            maes.append(mae)
            
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.plot(times, maes, marker="*")
        ax.set_title("Mean Absolute Error of Actual vs Predicted Wind Speed", fontsize=FONT_SIZE_TITLE)
        ax.set_ylabel("Mean Absolute Error (m/s)", fontsize=FONT_SIZE_AXES)
        ax.set_xlabel("Hours into the future", fontsize=FONT_SIZE_AXES)
        ax.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
               
    city_selection = widgets.Dropdown(
        options=weather_forecasts.keys(),
        description='City',
    )
    
    interact(_plot, city=city_selection)