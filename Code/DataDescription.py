import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
# import tensorflow as tf


class StockDataProcessor:
    _row_data = None
    _data_file_path = None
    _df : pd.DataFrame = None
    _features : list = []
    _target = "ret"


    def get_df(self):
        if self._df is None:
            raise ValueError("Dataframe is not initialized.")
        return self._df
    
    def get_features(self):        
        if self._features is None:
            raise ValueError("Features list is not initialized.")
        return self._features
    
    def get_row_data(self):
        if self._row_data is None:
            raise ValueError("Row data is not initialized.")
        return self._row_data

    def __init__(self, data_file_path):
        self._read_data(data_file_path)
        self._preprocess()

    def _read_data(self, data_file_path):
        self._row_data = pd.read_csv(data_file_path, encoding='utf-8', low_memory=False)
        self._df = pd.DataFrame(self._row_data)


    def _preprocess(self):
        # set all the columns' characters to lowercase
        self._set_lowercase()

        # 数据里面日期的格式比较特殊，这里转换成了标准日期格式
        self._change_date_type()

        # change to hot-encoding
        label_encoder = LabelEncoder()
        for column in self._df.select_dtypes(include=['object']).columns:
            self._df[column] = label_encoder.fit_transform(self._df[column])

        

    # set all the columns' characters to lowercase
    def _set_lowercase(self):
        self._df.columns = self._df.columns.astype(str).str.lower()
        self._target = "ret"
        self._features = self._df.drop(columns=[self._target]).columns.to_list()
        if 'date' in self._features:
            self._features.remove('date')

    def _change_date_type(self):
        # change the data type of 'date' to datetime
        self._df['date'] = pd.to_datetime(self._df['date'], format='%Y%m%d')
        self._df.dropna(subset=['date'], inplace=True)  # drop rows with missing date
        self._df.set_index('date', inplace=True)



def main():
    # 这是我的文件路径，在自己电脑跑记得改掉
    # for easier use, hard code in file path
    data_file_path = 'C:/Code/Python/FQA_final/Datas/GHZ_ZHY_V8.csv'
    data_processor = StockDataProcessor(data_file_path)
    
    df = data_processor.get_df()


    main_features = ['mom1m', 'mom12m', 'chmom', 'indmom', 'maxret', 'mom36m', 'turn', 'std_turn', 'mvel1', 'dolvol', 'zerotrade', 'baspread', 'retvol', 'idiovol', 'beta', 'betasq', 'ep', 'sp', 'agr', 'nincr']
    # get the descriptive statistics
    # descriptive_stats = df[main_features].describe().T     
    # print(descriptive_stats[['mean', 'std', 'min', 'max']])

    # # get the correlation matrix
    # correlation_matrix = df.corr()

    print("Plotting heatmap...")
    # 设置 Seaborn 样式
    # 计算特征之间的相关性矩阵
    correlation_matrix = df[main_features].corr()

    # 设置 Seaborn 样式
    sns.set(style="white")

    # 创建绘图窗口
    plt.figure(figsize=(12, 10))

    # 绘制相关性热力图
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, square=True, cbar_kws={'shrink': 0.8})
    # 设置标题
    plt.title('Main Features Correlation Heatmap of Features')

    # 保存图像而不是显示
    plt.savefig("main_features_correlation_heatmap.png")

    # 显示图像
    plt.show()




# main function
if __name__ == "__main__":
    main()
