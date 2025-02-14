# 导入必要的库  
import numpy as np  
import pandas as pd  
import seaborn as sns  
import ipywidgets as widgets  
import matplotlib.pyplot as plt  
import torch  
from torch import nn  
from torch.utils.data import DataLoader, TensorDataset  
import shap  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn import metrics  
from sklearn.inspection import permutation_importance  
from datetime import datetime, timedelta  
from ipywidgets import interact, interact_manual, fixed  
from typing import List, Iterable  

# 定义全局字体大小，用于绘图  
FONT_SIZE_TICKS = 15  # 坐标轴刻度字体大小  
FONT_SIZE_TITLE = 20  # 图表标题字体大小  
FONT_SIZE_AXES = 20   # 坐标轴标签字体大小  


def fix_temperatures(data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:  
    """  
    修复温度数据中异常值（过低温度），通过线性插值替换异常值。  

    参数:  
        data (pd.core.frame.DataFrame): 原始数据集。  

    返回:  
        pd.core.frame.DataFrame: 修复后的数据集。  
    """  
    # 找到环境温度（Etmp）中最低1%分位数的值，作为异常值的阈值  
    min_etemp = data["Etmp"].quantile(0.01)  
    # 将低于阈值的温度值替换为 NaN  
    data["Etmp"] = data["Etmp"].apply(lambda x: np.nan if x < min_etemp else x)  
    # 使用线性插值法填充 NaN 值  
    data["Etmp"] = data["Etmp"].interpolate()  

    # 同样处理内部温度（Itmp）  
    min_itemp = data["Itmp"].quantile(0.01)  
    data["Itmp"] = data["Itmp"].apply(lambda x: np.nan if x < min_itemp else x)  
    data["Itmp"] = data["Itmp"].interpolate()  

    return data  


def tag_abnormal_values(  
    df: pd.core.frame.DataFrame, condition: pd.core.series.Series  
) -> pd.core.frame.DataFrame:  
    """  
    根据条件标记数据集中是否存在异常值。  

    参数:  
        df (pd.core.frame.DataFrame): 数据集。  
        condition (pd.core.series.Series): 一个布尔类型的 Series，表示记录是否满足异常条件。  

    返回:  
        pd.core.frame.DataFrame: 标记了异常值的数据集。  
    """  
    # 找到满足条件的记录索引  
    indexes = df[condition].index  
    # 将这些记录的 "Include" 列标记为 False（表示异常）  
    df.loc[indexes, "Include"] = False  
    return df  


def cut_pab_features(raw_data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:  
    """  
    删除数据集中冗余的 Pab 特征，并重命名 Pab1 为 Pab。  

    参数:  
        raw_data (pd.core.frame.DataFrame): 原始数据集。  

    返回:  
        pd.core.frame.DataFrame: 删除冗余特征后的数据集。  
    """  
    # 删除 Pab2 和 Pab3 列  
    raw_data = raw_data.drop(["Pab2", "Pab3"], axis=1)  
    # 将 Pab1 列重命名为 Pab  
    raw_data = raw_data.rename(columns={"Pab1": "Pab"})  

    return raw_data  


def generate_time_signals(raw_data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:  
    """  
    为数据集生成时间信号特征（一天中的时间周期性特征）。  

    参数:  
        raw_data (pd.core.frame.DataFrame): 原始数据集。  

    返回:  
        pd.core.frame.DataFrame: 添加了时间信号特征的数据集。  
    """  
    # 如果已经存在 "Day sin" 列，则直接返回数据集  
    if "Day sin" in raw_data.columns:  
        return raw_data  

    # 将 Datetime 列转换为时间戳格式  
    date_time = pd.to_datetime(raw_data.Datetime, format="%Y-%m-%d %H:%M")  
    timestamp_s = date_time.map(pd.Timestamp.timestamp)  

    # 一天的秒数  
    day = 24 * 60 * 60  

    # 计算时间的正弦和余弦值，表示一天中的时间周期  
    raw_data["Time-of-day sin"] = np.sin(timestamp_s * (2 * np.pi / day))  
    raw_data["Time-of-day cos"] = np.cos(timestamp_s * (2 * np.pi / day))  

    return raw_data  


def top_n_turbines(  
    raw_data: pd.core.frame.DataFrame, n: int  
) -> pd.core.frame.DataFrame:  
    """  
    筛选发电量平均值最高的前 n 个风机的数据。  

    参数:  
        raw_data (pd.core.frame.DataFrame): 原始数据集。  
        n (int): 保留的风机数量。  

    返回:  
        pd.core.frame.DataFrame: 仅包含前 n 个风机的数据集。  
    """  
    # 按风机 ID 分组，计算发电量（Patv）的平均值，并排序 
    raw_data["Patv"] = pd.to_numeric(raw_data["Patv"], errors="coerce")
    sorted_patv_by_turbine = raw_data.groupby("TurbID")["Patv"].mean().sort_values(ascending=False)

    # 获取发电量最高的前 n 个风机的 ID  
    top_turbines = list(sorted_patv_by_turbine.index)[:n]
    # 打印原始数据的信息  
    print(  
        f"Original data has {len(raw_data)} rows from {len(raw_data.TurbID.unique())} turbines.\n"  
    )  

    # 筛选出前 n 个风机的数据  
    raw_data = raw_data[raw_data["TurbID"].isin(top_turbines)]  

    # 打印筛选后的数据信息  
    print(  
        f"Sliced data has {len(raw_data)} rows from {len(raw_data.TurbID.unique())} turbines."  
    )  

    return raw_data 


def format_datetime(  
    data: pd.core.frame.DataFrame, 
    initial_date_str: str  
) -> pd.core.frame.DataFrame:  
    """  
    将 Day 和 Tmstamp 列格式化为 Datetime 列。  

    参数:  
        data (pd.core.frame.DataFrame): 原始数据集。  
        initial_date_str (str): 初始日期的字符串格式。  

    返回:  
        pd.core.frame.DataFrame: 格式化后的数据集。  
    """  
    # 如果已经存在 "Datetime" 列，则直接返回数据集  
    if "Datetime" in data.columns:  
        return data  

    # 将初始日期字符串转换为日期对象  
    initial_date = datetime.strptime(initial_date_str, "%d %m %Y").date()  

    # 计算每一行的日期  
    data["Date"] = data.apply(  
        lambda x: str(initial_date + timedelta(days=(x.Day - 1))), axis=1  
    )  

    # 将日期和时间戳拼接成完整的日期时间格式  
    data["Datetime"] = data.apply(  
        lambda x: datetime.strptime(f"{x.Date} {x.Tmstamp}", "%Y-%m-%d %H:%M"), axis=1  
    )  

    # 删除原始 Day、Tmstamp 和中间生成的 Date 列  
    data.drop(["Day", "Tmstamp", "Date"], axis=1, inplace=True)  

    # 将 Datetime 列放在数据集的第一列  
    data = data[["Datetime"] + [col for col in list(data.columns) if col != "Datetime"]]  

    return data  


def transform_angles(  
    data: pd.core.frame.DataFrame, feature: str, drop_original: bool = True  
):  
    """  
    将角度特征转换为正弦和余弦特征。  

    参数:  
        data (pd.core.frame.DataFrame): 数据集。  
        feature (str): 角度特征的列名。  
        drop_original (bool, optional): 是否删除原始角度列。默认为 True。  
    """  
    # 将角度转换为弧度  
    rads = data[feature] * np.pi / 180  

    # 计算角度的正弦和余弦值  
    data[f"{feature}Cos"] = np.cos(rads)  
    data[f"{feature}Sin"] = np.sin(rads)  

    # 如果需要，删除原始角度列  
    if drop_original:  
        data.drop(feature, axis=1, inplace=True)  


def plot_wind_speed_vs_power(  
    ax: plt.Axes,  
    x1: Iterable,  
    y1: Iterable,  
    x2: Iterable,  
    y2: Iterable  
):  
    """  
    绘制风速与发电量的散点图。  

    参数:  
        ax (plt.Axes): 用于绘图的坐标轴。  
        x1, y1: 原始数据的 x 和 y 值。如果没有可以设置为 None。  
        x2, y2: 模型预测的 x 和 y 值。如果没有可以设置为 None。  
    """  
    # 绘制原始数据的散点图  
    ax.scatter(  
        x1, y1, color="blue", edgecolors="white", s=15, label="actual"  
    )  
    # 绘制模型预测数据的散点图  
    ax.scatter(  
        x2, y2,  
        color="orange", edgecolors="black", s=15, marker="D", label="model"  
    )  
    # 设置坐标轴标签和标题  
    ax.set_xlabel("Wind Speed (m/s)", fontsize=FONT_SIZE_AXES)  
    ax.set_ylabel("Active Power (kW)", fontsize=FONT_SIZE_AXES)  
    ax.set_title("Wind Speed vs. Power Output", fontsize=FONT_SIZE_TITLE)  
    ax.tick_params(labelsize=FONT_SIZE_TICKS)  
    ax.legend(fontsize=FONT_SIZE_TICKS)


def plot_predicted_vs_real(  
    ax: plt.Axes,  
    x1: Iterable,  
    y1: Iterable,  
    x2: Iterable,  
    y2: Iterable  
):  
    """  
    绘制预测值 vs 实际值的散点图。  

    参数:  
        ax (plt.Axes): 用于绘制的坐标轴对象。  
        x1, y1: 原始数据 x 和 y 值（用于绘点）。可以为空。  
        x2, y2: 拟合线的 x 和 y 值（用于绘制直线）。可以为空。  
    """  
    # 绘制预测值与实际值的散点图  
    ax.scatter(  
        x1, y1, color="orange", edgecolors="black", label="Predicted vs. actual values", marker="D"  
    )  
    # 绘制理想情况（预测值 = 实际值）的直线  
    ax.plot(  
        x2, y2, color="blue", linestyle="--", linewidth=4, label="actual = predicted",  
    )  
    # 设置坐标轴标签与标题  
    ax.set_xlabel("Actual Power Values (kW)", fontsize=FONT_SIZE_AXES)  
    ax.set_ylabel("Predicted Power Values (kW)", fontsize=FONT_SIZE_AXES)  
    ax.set_title("Predicted vs. Actual Power Values (kW)", fontsize=FONT_SIZE_TITLE)  
    ax.tick_params(labelsize=FONT_SIZE_TICKS)  
    ax.legend(fontsize=FONT_SIZE_TICKS)  


def fit_and_plot_linear_model(data_og: pd.core.frame.DataFrame, turbine: int, features: List[str]):  
    """  
    拟合线性模型并绘制预测结果图。  

    参数:  
        data_og (pd.core.frame.DataFrame): 原始数据集。  
        turbine (int): 风机 ID。  
        features (List[str]): 用于预测的特征列表。  
    """  
    # 获取选定风机的数据  
    data = data_og[data_og.TurbID == turbine]  

    # 创建线性回归模型  
    features = list(features)  
    y = data["Patv"]  # 输出变量（功率）  
    X = data[features]  # 输入变量（特征）  
    # 拆分为训练集和测试集  
    X_train, X_test, y_train, y_test = train_test_split(  
        X, y, test_size=0.2, random_state=42  
    )  
    reg = LinearRegression().fit(X_train, y_train)  

    # 准备左图（风速 vs 功率）的数据  
    X_plot = data["Wspd"]  
    Y_real = data["Patv"]  
    y_test_preds = reg.predict(X_test)  

    # 准备右图（预测 vs 实际值）的数据  
    X_eq_Y = np.linspace(0, max([max(y_test), max(y_test_preds)]), 100)  

    # 创建两个子图来分别展示  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  
    # 左侧绘图：风速 vs 功率  
    if "Wspd" in features:  
        plot_wind_speed_vs_power(ax1, X_plot, Y_real, X_test["Wspd"], y_test_preds)  
    else:  
        plot_wind_speed_vs_power(ax1, X_plot, Y_real, None, None)  
        print("The model could not be plotted here as Wspd is not among the features")  
    # 右侧绘图：预测值 vs 实际值  
    plot_predicted_vs_real(ax2, y_test, y_test_preds, X_eq_Y, X_eq_Y)  

    plt.tight_layout()  
    plt.show()  

    # 如果有多于一个特征，则计算并绘制特征重要性  
    if len(features) > 1:  
        # 计算特征的重要性分数  
        bunch = permutation_importance(  
            reg, X_test, y_test, n_repeats=10, random_state=42  
        )  
        imp_means = bunch.importances_mean  
        ordered_imp_means_args = np.argsort(imp_means)[::-1]  # 按重要性降序排序  

        results = {}  
        for i in ordered_imp_means_args:  
            name = list(X_test.columns)[i]  
            imp_score = imp_means[i]  
            results.update({name: [imp_score]})  

        results_df = pd.DataFrame.from_dict(results)  

        # 绘制特征重要性  
        fig, ax = plt.subplots(figsize=(7.5, 6))  
        ax.set_xlabel("Importance Score", fontsize=FONT_SIZE_AXES)  
        ax.set_ylabel("Feature", fontsize=FONT_SIZE_AXES)  
        ax.set_title("Feature Importance", fontsize=FONT_SIZE_TITLE)  
        ax.tick_params(labelsize=FONT_SIZE_TICKS)  

        sns.barplot(data=results_df, orient="h", ax=ax, color="deepskyblue", width=0.3)  

        plt.show()  

    # 打印平均绝对误差 (MAE)  
    mae = metrics.mean_absolute_error(y_test, y_test_preds)  
    print(f"Turbine {turbine}, Mean Absolute Error (kW): {mae:.2f}\n")  


def linear_univariate_model(data_og: pd.core.frame.DataFrame):  
    """  
    创建基于风速的单变量线性模型，并生成交互式可视化。  

    参数:  
        data_og (pd.core.frame.DataFrame): 数据集。  
    """  
    # 创建风机选择器  
    turbine_selection = widgets.Dropdown(  
        options=data_og.TurbID.unique(), description="Turbine"  
    )  

    # 通过交互式组件展示单变量线性回归模型  
    interact(fit_and_plot_linear_model, data_og=fixed(data_og), turbine=turbine_selection, features=fixed(["Wspd"]))  


def linear_multivariate_model(data_og: pd.core.frame.DataFrame, features: List[str]):  
    """  
    创建基于多变量的线性模型，并生成交互式可视化。  

    参数:  
        data_og (pd.core.frame.DataFrame): 数据集。  
        features (List[str]): 用于预测的特征列表。  
    """  
    # 创建风机选择器  
    turbine_selection = widgets.Dropdown(  
        options=data_og.TurbID.unique(), description="Turbine"  
    )  

    # 创建多选特征组件  
    feature_selection = widgets.SelectMultiple(  
        options=features,  
        value=list(features),  
        description="Features",  
        disabled=False,  
    )  

    # 创建交互式模型可视化  
    interact_manual(fit_and_plot_linear_model, data_og=fixed(data_og),  
                    turbine=turbine_selection, features=feature_selection)  


def split_and_normalize(data: pd.core.frame.DataFrame, features: List[str]):  
    """  
    生成训练集与测试集，并对数据进行标准化（均值归零，单位归一化）。  

    参数:  
        data (pd.core.frame.DataFrame): 原始数据集。  
        features (List[str]): 用于预测的特征。  

    返回:  
        tuple: 标准化后的训练集/测试集及相关统计信息（均值与标准差）。  
    """  
    X = data[features]  
    y = data["Patv"]  
    # 将数据拆分为训练集和测试集  
    X_train, X_test, y_train, y_test = train_test_split(  
        X, y, test_size=0.2, random_state=42  
    )  
    to_normalize = ["Wspd", "Etmp", "Itmp", "Prtv"]  

    f_to_normalize = [feature for feature in features if feature in to_normalize]  
    f_not_to_normalize = [  
        feature for feature in features if feature not in to_normalize  
    ]  

    # 计算用于归一化的均值与标准差  
    X_train_mean = X_train[f_to_normalize].mean()  
    X_train_std = X_train[f_to_normalize].std()  

    y_train_mean = y_train.mean()  
    y_train_std = y_train.std()  

    # 标准化数值  
    X_train[f_to_normalize] = (X_train[f_to_normalize] - X_train_mean) / X_train_std  
    X_test[f_to_normalize] = (X_test[f_to_normalize] - X_train_mean) / X_train_std  

    y_train = (y_train - y_train_mean) / y_train_std  
    y_test = (y_test - y_train_mean) / y_train_std  

    # 转换为 NumPy 数组并进一步转换为 Torch 张量  
    X_train = torch.from_numpy(X_train.to_numpy()).type(torch.float)  
    X_test = torch.from_numpy(X_test.to_numpy()).type(torch.float)  
    y_train = torch.from_numpy(y_train.to_numpy()).type(torch.float).unsqueeze(dim=1)  
    y_test = torch.from_numpy(y_test.to_numpy()).type(torch.float).unsqueeze(dim=1)  

    return (X_train, X_test, y_train, y_test), (  
        X_train_mean,  
        X_train_std,  
        y_train_mean,  
        y_train_std,  
    )  


def batch_data(  
    X_train: torch.Tensor,  
    X_test: torch.Tensor,  
    y_train: torch.Tensor,  
    y_test: torch.Tensor,  
    batch_size: int,  
):  
    """  
    根据批量大小创建训练集和测试集的数据加载器。  

    参数:  
        X_train (torch.Tensor): 训练集特征。  
        X_test (torch.Tensor): 测试集特征。  
        y_train (torch.Tensor): 训练集目标。  
        y_test (torch.Tensor): 测试集目标。  
        batch_size (int): 批量大小。  

    返回:  
        tuple: 训练集和测试集的数据加载器。  
    """  
    train_dataset = TensorDataset(X_train, y_train)  
    train_loader = DataLoader(train_dataset, batch_size=batch_size)  

    test_dataset = TensorDataset(X_test, y_test)  
    test_loader = DataLoader(test_dataset, batch_size=batch_size)  

    return train_loader, test_loader  


class RegressorNet(nn.Module):  
    """  
    简单的前馈神经网络回归模型, 包含3个隐藏层。  
    """  

    def __init__(self, input_size):  
        super().__init__()  

        self.fc_layers = nn.Sequential(  
            nn.Linear(input_size, 64),  
            nn.ReLU(),  
            nn.Linear(64, 32),  
            nn.ReLU(),  
            nn.Linear(32, 1),  
        )  

    def forward(self, x):  
        x = self.fc_layers(x)  
        return x  


def compile_model(features: List[str]):  
    """  
    定义 PyTorch 网络模型、损失函数和优化器。  

    参数:  
        features (List[str]): 用于预测的特征。  

    返回:  
        tuple: 模型、损失函数和优化器。  
    """  
    model = RegressorNet(input_size=len(features))  
    loss_fn = nn.L1Loss()  # 使用平均绝对误差作为损失函数  
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)  

    return model, loss_fn, optimizer  

def train_model(  
    model: RegressorNet,  
    loss_fn: torch.nn.modules.loss.L1Loss,  
    optimizer: torch.optim.Adam,  
    train_loader: torch.utils.data.DataLoader,  
    test_loader: torch.utils.data.DataLoader,  
    epochs: int,  
):  
    """  
    训练神经网络模型。  

    参数:  
        model (RegressorNet): 神经网络模型的实例。  
        loss_fn (torch.nn.modules.loss.L1Loss): L1 损失函数（即平均绝对误差）。  
        optimizer (torch.optim.Adam): Adam 优化器。  
        train_loader (torch.utils.data.DataLoader): 训练数据加载器。  
        test_loader (torch.utils.data.DataLoader): 测试数据加载器。  
        epochs (int): 训练的迭代次数。  

    返回:  
        RegressorNet: 已训练好的模型。  
    """  
    for epoch in range(epochs):  
        # 标记模型为训练模式  
        model.train()  

        for batch, (X, y) in enumerate(train_loader):  
            # 1. 前向传播  
            y_pred = model(X)  
            # 2. 计算损失  
            loss = loss_fn(y_pred, y)  
            # 3. 清除优化器中的梯度  
            optimizer.zero_grad()  
            # 4. 反向传播  
            loss.backward()  
            # 5. 更新模型参数  
            optimizer.step()  

        # 测试模型性能  
        model.eval()  
        with torch.inference_mode():  # 关闭梯度计算以提高推理性能  
            for batch, (X, y) in enumerate(test_loader):  
                # 1. 前向传播  
                test_pred = model(X)  

                # 2. 计算测试损失  
                test_loss = loss_fn(test_pred, y)  

        # 每隔一个周期打印训练和测试损失  
        if epoch % 1 == 0:  
            print(  
                f"Epoch: {epoch} | Train loss: {loss:.5f} | Test loss: {test_loss:.5f}"  
            )  

    return model  


def plot_feature_importance(  
    model: RegressorNet,  
    features: List[str],  
    train_loader: torch.utils.data.DataLoader,  
    test_loader: torch.utils.data.DataLoader,  
):  
    """  
    使用 SHAP 值分析绘制特征重要性图。  

    参数:  
        model (RegressorNet): 已训练的模型。  
        features (List[str]): 用于预测的特征列表。  
        train_loader (torch.utils.data.DataLoader): 训练数据加载器。  
        test_loader (torch.utils.data.DataLoader): 测试数据加载器。  
    """  
    # 获取训练和测试的一个批次数据  
    x_train_batch, _ = next(iter(train_loader))  
    x_test_batch, _ = next(iter(test_loader))  

    # 设置模型为评估模式  
    model.eval()  

    # 使用 SHAP DeepExplainer 计算特征重要性  
    e = shap.DeepExplainer(model, x_train_batch)  
    shap_values = e.shap_values(x_test_batch)  
    
    # 计算每个特征的重要性得分  
    means = np.mean(np.abs(shap_values), axis=0)  # 取绝对值的平均值  
    results = sorted(zip(features, means), key=lambda x: x[1], reverse=True)  
    results_df = pd.DataFrame.from_dict({k: [v] for (k, v) in results})  

    # 绘制特征重要性  
    fig, ax = plt.subplots(figsize=(7.5, 6))  
    ax.set_xlabel("Importance Score", fontsize=FONT_SIZE_AXES)  
    ax.set_ylabel("Feature", fontsize=FONT_SIZE_AXES)  
    ax.set_title("Feature Importance", fontsize=FONT_SIZE_TITLE)  
    ax.tick_params(labelsize=FONT_SIZE_TICKS)  
    sns.barplot(data=results_df, orient="h", ax=ax, color="deepskyblue", width=0.3)  
    
    return shap_values  


def neural_network(data_og: pd.core.frame.DataFrame, features: List[str]):  
    """  
    创建神经网络预测过程的交互式图表。  

    参数:  
        data_og (pd.core.frame.DataFrame): 原始数据。  
        features (List[str]): 用于预测的特征。  
    """  

    def fit_nn(turbine, features):  
        """  
        训练神经网络并显示预测结果。  

        参数:  
            turbine: 所选风机 ID。  
            features: 用于训练的特征。  
        """  
        # 筛选指定风机的数据  
        data = data_og[data_og.TurbID == turbine]  
        features = list(features)  

        print(f"Features used: {features}\n")  
        print(f"Training your Neural Network...\n")  

        # 数据标准化与拆分  
        (X_train, X_test, y_train, y_test), (  
            X_train_mean,  
            X_train_std,  
            y_train_mean,  
            y_train_std,  
        ) = split_and_normalize(data, features)  
        # 创建批量数据加载器  
        train_loader, test_loader = batch_data(  
            X_train, X_test, y_train, y_test, batch_size=32  
        )  
        # 编译模型  
        model, loss_fn, optimizer = compile_model(features)  
        # 训练模型  
        model = train_model(  
            model, loss_fn, optimizer, train_loader, test_loader, epochs=5  
        )  

        print(f"\nResults:")  

        # 反归一化测试目标和预测值  
        y_test_denormalized = (y_test * y_train_std) + y_train_mean  
        test_preds = model(X_test).detach().numpy()  
        test_preds_denormalized = (test_preds * y_train_std) + y_train_mean  

        # 创建风速-功率散点图，以及预测值-实际值对比图  
        X_plot = data["Wspd"]  
        Y_real = data["Patv"]  
        X_eq_Y = np.linspace(0, max(y_test_denormalized), 100)  

        print(  
            f"Mean Absolute Error: {metrics.mean_absolute_error(y_test_denormalized, test_preds_denormalized):.2f}\n"  
        )  

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  
        
        if "Wspd" in features:  
            test_preds = model(X_test).detach().numpy()  
            test_preds_denormalized = (test_preds * y_train_std) + y_train_mean  

            # 将测试集的风速数据反归一化  
            X_test_2 = X_test.detach().numpy()  
            X_test_denormalized = (X_test_2[:, 0] * X_train_std[0]) + X_train_mean[0]  
            
            # 绘制左图：风速 vs 功率  
            plot_wind_speed_vs_power(ax1, X_plot, Y_real, X_test_denormalized, test_preds_denormalized)  
        else:  
            plot_wind_speed_vs_power(ax1, X_plot, Y_real, None, None)  
            print("The model could not be plotted here as Wspd is not among the features")  

        # 绘制右图：预测值 vs 实际值  
        plot_predicted_vs_real(ax2, y_test_denormalized, test_preds_denormalized, X_eq_Y, X_eq_Y)  

        plt.show()  

        # 使用 SHAP 绘制特征重要性  
        train_loader, test_loader = batch_data(  
            X_train, X_test, y_train, y_test, batch_size=128  
        )  
        plot_feature_importance(model, features, train_loader, test_loader)  

    # 创建交互式组件（风机选择器和特征选择器）  
    turbine_selection = widgets.Dropdown(  
        options=data_og.TurbID.unique(), description="Turbine"  
    )  
    feature_selection = widgets.SelectMultiple(  
        options=features,  
        value=list(features),  
        description="Features",  
        disabled=False,  
    )  
    interact_manual(fit_nn, turbine=turbine_selection, features=feature_selection)