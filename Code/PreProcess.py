import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
import numpy as np
# import tensorflow as tf

'''
    This class preprocesses the data and splits it into train, val, and test sets.
    How to use:

        data_processor = StockDataProcessor(data_file_path)

    every time you need to read a new data file, pass the data_file_path to the constructor.

    1. If you need to get the dataframe and add other preprocess steps, use the "get_df()" method.
    
    2. If you need to get the column list, use the "get_columnlist()" method.

    3. If you need to get the train, val, and test sets, use the "datasets" attributes,
    which are dictionaries with keys "train", "val", and "test".

        For example:

            data_processor = StockDataProcessor(data_file_path)
            train_data = data_processor.datasets["train"]

    4. If you want to get split X and y, call "split_dataset_with_target(df)", which returns X and y.

        For example:

            train_X, train_y = split_dataset_with_target(train_data)
        
'''

class StockDataProcessor:

    _row_data = None
    _data_file_path = None
    _df : pd.DataFrame = None
    _features : list = []
    _target = "ret"

    datasets = {"train": None, "val": None, "test": None}  # storing features
    # dataset_y = {"train": None, "val": None, "test": None}  # storing target column



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
    
    def save_data(self, file_path = None):
        if file_path is None:
            file_path = self._data_file_path
        self._df.to_csv(file_path, index=False)


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

        # print("num of columns: ", len(self._features), "\nColumn names:\n")
        # for i in range(0, len(self._features), 8):
        #     print(self._features[i:i+8])
        # print("\n")

        # change to hot-encoding
        label_encoder = LabelEncoder()
        for column in self._df.select_dtypes(include=['object']).columns:
            self._df[column] = label_encoder.fit_transform(self._df[column])

        # split the datasets
        self._split_data()

        # print and deal with missing values                  
        self._deal_with_missing_values()

        self.deal_with_outlier()

        # normalize the data
        self._normalize_data()

        

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

    
    def _deal_with_missing_values(self):
        # get missing values and print them
        # missing_values = self._df.isnull().sum()
        # pd.set_option('display.max_columns', None)
        # print("Missing values every column:\n", missing_values, "\n")


        # # Variable plt_feature stores the feature name for plotting, default is "beta"
        # plt_feature = "beta"
        # print("Input the feature name for plotting: ")
        # plt_feature = input()

        # # plot the feature distribution after processing
        # while plt_feature not in self._columns:
        #     print(f"Feature '{plt_feature}' not found in the DataFrame.")
        #     print("Input the feature name for plotting: ")
        #     plt_feature = input()

        # 对于正太分布且与时间无关的特征，下面采用均值法填充缺失值
        # fill missing values with mean
        mean_fill_features = ['bm', 'cfp', 'lev', 'agr', 'lgr', 'ps', 'quick', 'roaq', 'roeq', 'roic', 'sgr', 'tang']
        for feature in mean_fill_features:
            if feature not in self.datasets["train"].columns:
                print(f"Feature '{feature}' not found in the DataFrame.")
                continue
            mean_value = self._df[feature].mean()
            self._df[feature].fillna(mean_value)

        # 下列特征变化平稳，可以采用线性插值法填充缺失值
        # fill missing values with linear interpolation
        linear_features = ['beta', 'bm_ia', 'chinv', 'currat', 'cashdebt', 'depr', 'dy', 'ep', 'gma', 'rd_sale']
        for feature in linear_features:
            if feature not in self.datasets["train"].columns:
                print(f"Feature '{feature}' not found in the DataFrame.")
                continue
            self._df[feature] = self._df[feature].interpolate(method="linear")

        # 下列特征的变化与时间高度相关，采用时间序列插值法填充缺失值
        # fill missing values with time series interpolation
        time_series_features = ['mom1m', 'mom6m', 'mom12m', 'mom36m', 'std_turn', 'std_dolvol', 'turn', 'retvol', 'maxret', 'rsup']
        for feature in time_series_features:
            if feature not in self.datasets["train"].columns:
                print(f"Feature '{feature}' not found in the DataFrame.")
                continue
            self._df[feature] = self._df[feature].interpolate(method="time")

        # if you just want to replace missing values with 0, comment the above codes

        # 其余全部缺失值用0填充
        # replace all missing values with 0
        self.datasets["train"].fillna(0, inplace=True)

        # get missing values and print them again
        # missing_values = self._df.isnull().sum()
        # print("Missing values every column:\n", missing_values, "\n")


    def _split_data(self):
        # 按照时间划分数据，这里参考哈里斯的tutorial
        # 
        # datas will be split into training set, validation set and testing set by year
        # 1957-2004 for training set, 2005-2009 for validation set, 2010-2016 for testing set
        self._df['year'] = self._df.index.year
        ind_train = self._df[self._df.year.isin(range(1926,2005))] # 1957 to 2004
        ind_val = self._df[self._df.year.isin(range(2005,2010))] # 2005 to 2009
        ind_test = self._df[self._df.year.isin(range(2010,2017))] # 2010 to 2016

        train_data = ind_train.copy().reset_index(drop=True)
        val_data = ind_val.copy().reset_index(drop=True)
        test_data = ind_test.copy().reset_index(drop=True)

        # # get the target column
        # target_column = "ret"

        # self.dataset_X["train"], self.dataset_y["train"] = split_dataset_with_target(train_data, target_column)
        # self.dataset_X["val"], self.dataset_y["val"] = split_dataset_with_target(val_data, target_column)
        # self.dataset_X["test"], self.dataset_y["test"] = split_dataset_with_target(test_data, target_column)
        self.datasets["train"] = train_data
        self.datasets["val"] = val_data
        self.datasets["test"] = test_data
   
    def _normalize_data(self):
        for key in ["train", "val", "test"]:
            self.datasets[key][self._features] = self.datasets[key][self._features].apply(normalize).fillna(0)
            self.datasets[key] = self.datasets[key][self._features + [self._target]]  

    def deal_with_outlier(self):
        # 确保训练集数据不为空
        if self.datasets["train"].empty:
            print("Training dataset is empty. Skipping outlier removal.")
            return

        print("Size before deal_with_outlier:", len(self.datasets["train"]))

        # 识别连续特征（float 类型，且唯一值数目多）
        continuous_features = [
            feature for feature in self._features
            if self.datasets["train"][feature].dtype == 'float64' and self.datasets["train"][feature].nunique() > 10
        ]

        # 初始化掩码
        mask = pd.Series(True, index=self.datasets["train"].index)

        # 遍历连续特征进行异常值处理
        for feature in continuous_features:
            mean = self.datasets["train"][feature].mean()
            std = self.datasets["train"][feature].std()

            # 如果标准差为 0，跳过该特征
            if std == 0:
                print(f"Skipping feature {feature} with zero standard deviation.")
                continue

            # 计算掩码，仅保留 mean ± 15*std 范围内的样本
            feature_mask = (self.datasets["train"][feature] >= mean - 15 * std) & (self.datasets["train"][feature] <= mean + 15 * std)

            # 合并掩码
            mask &= feature_mask

        # 应用掩码
        self.datasets["train"] = self.datasets["train"][mask]

        print(f"Removed {len(mask) - mask.sum()} rows with outliers from training data.")
        print("Size after deal_with_outlier:", len(self.datasets["train"]))

    def save_datasets_to_csv(self, train_path, val_path, test_path):
        self.datasets["train"].to_csv(train_path, index=True)
        self.datasets["val"].to_csv(val_path, index=True)
        self.datasets["test"].to_csv(test_path, index=True)




def normalize(series):
  return (series-series.mean(axis=0))/series.std(axis=0)


def split_dataset_with_target(df):
    X = df.drop(columns=["ret"])
    y = df["ret"]
    return X, y

def ks_normality_test(df):
    normal_features = []
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        stat, p_value = stats.kstest(df[column].dropna(), 'norm')  # 检验正态性
        if p_value > 0.05:
            normal_features.append(column)
    return normal_features



def main():
    # 这是我的文件路径，在自己电脑跑记得改掉
    # for easier use, hard code in file path
    data_file_path = 'C:/Code/Python/FQA_final/Datas/GHZ_ZHY_V8.csv'
    data_processor = StockDataProcessor(data_file_path)

    train_path = 'C:/Code/Python/FQA_final/Datas/train_data.csv'
    val_path = 'C:/Code/Python/FQA_final/Datas/val_data.csv'
    test_path = 'C:/Code/Python/FQA_final/Datas/test_data.csv'

    data_processor.save_datasets_to_csv(train_path, val_path, test_path)




# main function
if __name__ == "__main__":
    main()
