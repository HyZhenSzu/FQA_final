# -*- coding = utf-8 -*-
# @Time     : 2024/12/12 12:13
# @Author   : Yao Jiamin
# @File     : Assignment4.py
# @Software : PyCharm

import pandas as pd
import numpy as np
import tensorflow as tf
import PreProcess
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
import seaborn as sns


def plot_all_models_feature_importance_heatmap_fixed(all_feature_importances, feature_names,
                                                     output_path="all_models_feature_importance_heatmap_fixed.png"):
    # 将所有模型的特征重要性合并成 DataFrame
    importance_df = pd.DataFrame(all_feature_importances, columns=feature_names,
                                 index=[f"NN{i}" for i in range(1, len(all_feature_importances) + 1)])

    # 转置 DataFrame，让特征为行，模型为列
    importance_df = importance_df.T

    # 设置更大的画布大小，以适应特征数量
    plt.figure(figsize=(12, max(8, len(feature_names) * 0.2)))  # 调整高度：每个特征分配0.2个单位高度
    sns.heatmap(importance_df, annot=False, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title("Feature Importances for NN1-NN5 (Heatmap - Transposed)")
    plt.tight_layout()

    # 保存热力图
    plt.savefig(output_path)
    plt.close()


# 绘制前20个重要特征的特征重要性柱状图
def plot_feature_importance(features, importances, layer_num, output_path="feature_importance.png"):
    # Sort features by importance
    feature_importance = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    top_features = feature_importance[:20]  # Select top 20 features

    # Split into names and values
    feature_names, importance_values = zip(*top_features)

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importance_values, color=(0.2588, 0.5294, 0.9608))
    plt.xlabel("Importance")
    plt.title(f"NN{layer_num} - Top 20 Feature Importances")
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important on top
    plt.tight_layout()

    # Save the chart
    plt.savefig(output_path)
    plt.close()


# Calculate feature importance from model weights
def calculate_feature_importance(model, feature_names):
    first_layer_weights = model.layers[0].get_weights()[0]  # Get weights of the first Dense layer
    feature_importance = np.mean(np.abs(first_layer_weights), axis=1)  # Average absolute weights
    return feature_importance


# 自定义 R2 损失函数
def r2_loss(y_true, y_pred):
    """
    R^2 Loss Function
    Loss = 1 - R^2 = Residual Sum of Squares / Total Sum of Squares
    """
    residual_sum_of_squares = K.sum(K.square(y_true - y_pred))
    total_sum_of_squares = K.sum(K.square(y_true - K.mean(y_true)))
    r2 = 1 - residual_sum_of_squares / (total_sum_of_squares + K.epsilon())
    return 1 - r2  # 损失是 1 - R^2

class MetricsPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, output_path):
        super().__init__()
        self.validation_data = validation_data
        self.output_path = output_path
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.val_r2 = []
        self.val_mse = []

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch + 1)
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_mse.append(logs.get('val_mse'))

        X_val, y_val = self.validation_data
        y_pred = self.model.predict(X_val, verbose=0)
        r2 = R2(y_val, y_pred.flatten())
        self.val_r2.append(r2)

    def on_train_end(self, logs=None):
        plt.figure(figsize=(12, 8))

        # Plot Loss
        plt.subplot(3, 1, 1)
        plt.plot(self.epochs, self.train_loss, label='Train Loss', color='blue')
        plt.plot(self.epochs, self.val_loss, label='Validation Loss', color='orange')
        plt.legend()
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        # Plot MSE
        plt.subplot(3, 1, 2)
        plt.plot(self.epochs, self.val_mse, label='Validation MSE', color='red')
        plt.legend()
        plt.title("Mean Squared Error (MSE)")
        plt.xlabel("Epochs")
        plt.ylabel("MSE")

        # Plot R2
        plt.subplot(3, 1, 3)
        plt.plot(self.epochs, self.val_r2, label='Validation R2', color='green')
        plt.legend()
        plt.title("R2 Score")
        plt.xlabel("Epochs")
        plt.ylabel("R2")

        # Save the final plot
        plt.tight_layout()
        plt.savefig(self.output_path)
        plt.close()

# Build model
def build_model(hid_no, node_num, l2_strength=0.01, dropout_rate=0.2):
    layers_statement = [
        layers.Dense(
            node_num,
            activation='relu',
            input_shape=[nfeats],
            kernel_regularizer=regularizers.l2(l2=l2_strength)
        ),
        layers.Dropout(rate=dropout_rate)
    ]
    for i in range(1, hid_no):
        node_num = int(node_num / 2)
        layers_statement.append(
            layers.Dense(
                node_num,
                activation='relu',
                kernel_regularizer=regularizers.l2(l2=l2_strength)
            )
        )
        layers_statement.append(layers.Dropout(rate=dropout_rate))
    layers_statement.append(
        layers.Dense(
            1,
            kernel_regularizer=regularizers.l2(l2=l2_strength)
        )
    )

    model = keras.Sequential(layers_statement)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=0.01,  # 初始学习率
    #     decay_steps=10000,  # 每 10000 步衰减一次
    #     decay_rate=0.9  # 学习率衰减因子
    # )
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


    model.compile(
        loss=r2_loss,
        optimizer=optimizer,
        metrics=['mse', r2_loss]
    )
    return model

# Calculate R2
def R2(y, y_hat):
    R2 = 1 - np.sum((y - y_hat)**2) / np.sum(y**2)
    return R2

if __name__ == '__main__':
    file_path = r"E:\\HyZhen\\pythonProject\\FinancialAnalysis\\Assignment4\\GHZ_ZHY_V8(1).csv"
    num_of_1st_layer_node = 32
    l2_strength = 0.0001
    np.random.seed(24)

    data_processor = PreProcess.StockDataProcessor(file_path)
    feats = data_processor.get_features()
    target = 'ret'

    df_train = data_processor.datasets["train"]
    df_test = data_processor.datasets["test"]
    data_train = df_train[feats].values
    data_test = df_test[feats].values

    scaler = StandardScaler()
    data_train = scaler.fit_transform(data_train)
    df_train[target] = (df_train[target] - df_train[target].mean()) / df_train[target].std()

    train_dataset = tf.data.Dataset.from_tensor_slices((data_train, df_train[target].values))
    test_dataset = tf.data.Dataset.from_tensor_slices((data_test, df_test[target].values))

    nfeats = len(feats)
    max_n_hid = 5
    all_feature_importances = []

    for hid_no in range(1, max_n_hid + 1):  # 从 NN1 到 NN5
        print(f"Starts NN{hid_no} model fitting with L2 regularization (strength={l2_strength})")
        model = build_model(hid_no, num_of_1st_layer_node, l2_strength=l2_strength, dropout_rate=0.2)
        model.summary()

        val_data = (data_test, df_test[target].values)

        output_file = f"NN{hid_no}_metrics_plot.png"
        metrics_plot_callback = MetricsPlotCallback(validation_data=val_data, output_path=output_file)

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            min_delta=1e-4
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )

        model.fit(
            train_dataset.batch(32),
            epochs=200,
            validation_data=val_data,
            callbacks=[metrics_plot_callback, early_stopping, reduce_lr]
        )

        # 计算特征重要性
        feature_importances = calculate_feature_importance(model, feats)

        # 存储每个模型的特征重要性
        all_feature_importances.append(feature_importances)

        # # 绘制特征重要性条形图
        # feature_importance_path = f"NN{hid_no}_feature_importance.png"
        # plot_feature_importance(
        #     features=feats,
        #     importances=feature_importances,
        #     layer_num=hid_no,
        #     output_path=feature_importance_path
        # )
        # print(f"Feature importance bar chart saved for NN{hid_no} at {feature_importance_path}")


        test_predictions = model.predict(test_dataset.batch(1)).flatten()
        R2_Val = R2(df_test[target].values, test_predictions)
        print(f"R2_Val for NN{hid_no}:{R2_Val}")
        print(f"End NN{hid_no} model fitting")

    # 所有模型训练完后，绘制五个模型的特征重要性热力图
    plot_all_models_feature_importance_heatmap_fixed(
        all_feature_importances=all_feature_importances,
        feature_names=feats,
        output_path="all_models_feature_importance_heatmap_fixed.png"
    )
    print(
        "Feature importance heatmap (transposed) for all models saved at all_models_feature_importance_heatmap_fixed.png")
