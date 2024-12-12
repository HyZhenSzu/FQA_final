import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf


class StockDataProcessor:

    _row_data = None
    _df : pd.DataFrame = None
    _columns : list = []

    dataset_X = {"train": None, "val": None, "test": None}  # storing features
    dataset_y = {"train": None, "val": None, "test": None}  # storingtarget column


    def get_df(self):
        return self._df
    
    def get_columns(self):
        return self._columns
    
    def get_row_data(self):
        return self._row_data
    
    def save_data(self, file_path):
        self._df.to_csv(file_path, index=False)

    
    def preprocess(self):
        # set all the columns' characters to lowercase
        self._set_lowercase()

        # 数据里面日期的格式比较特殊，这里转换成了标准日期格式
        self._change_date_type()

        # get the column names and print them
        columns = self.get_df().columns.tolist()

        # 所有特征名，直接打出来看着乱乱的，我就拆成一行8个了
        # 不想看特征名输出要从这下面的代码开始注释
        print("num of columns: ", len(columns))
        for i in range(0, len(columns), 8):
            print(columns[i:i+8])
        print("\n")

        # delete the useless features
        self._df.drop(columns=['mom12m'], inplace=True)
        self._df.drop(columns=['mom36m'], inplace=True)
        # delete the features which std over 15
        self.filter_features_by_std(threshold = 15)

        # print and deal with missing values                  
        self._deal_with_missing_values()

        # split the datasets
        self._split_data()


    def __init__(self, data_file_path):
        self._row_data = pd.read_csv(data_file_path, encoding='utf-8', low_memory=False)
        self._df = pd.DataFrame(self._row_data)

    # set all the columns' characters to lowercase
    def _set_lowercase(self):
        self._df.columns = self._df.columns.str.lower()
        self._columns = self._df.columns.tolist()

    def _change_date_type(self):
        # change the data type of 'date' to datetime
        self._df['date'] = pd.to_datetime(self._df['date'], format='%Y%m%d')
        self._df.set_index('date', inplace=True)

    # delete the features which std over threshold
    def filter_features_by_std(self, threshold):
        
        numeric_features = self._df.select_dtypes(include=[np.number]) 
        numeric_features = numeric_features.drop(columns=['ret'], errors='ignore')
        print("Numeric Features:\n", numeric_features)

        stds = numeric_features.std()
        selected_features = stds[stds <= threshold].index
        self.filtered_data = self._df[selected_features.to_list() + ['ret']]
        self._columns = selected_features.to_list() + ['ret']
        print("num of columns after filtering: ", len(self._columns))

    
    def _deal_with_missing_values(self):
        # get missing values and print them
        missing_values = self._df.isnull().sum()
        pd.set_option('display.max_columns', None)
        print("Missing values every column:\n", missing_values, "\n")


        # Variable plt_feature stores the feature name for plotting, default is "beta"
        plt_feature = "beta"
        print("Input the feature name for plotting: ")
        plt_feature = input()

        # plot the feature distribution after processing
        while plt_feature not in self._columns:
            print(f"Feature '{plt_feature}' not found in the DataFrame.")
            print("Input the feature name for plotting: ")
            plt_feature = input()

        plt.figure(figsize=(6, 4))
        plt.hist(self._df[plt_feature], bins=30, color="green", alpha=0.7, label="Data (Before Processing)")
        plt.title(f"{plt_feature}: Distribution After Processing")
        plt.xlabel(plt_feature)
        plt.ylabel("Frequency")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.legend()
        plt.show()
            

        # 对于分布稳定且与时间无关的特征，下面采用均值法填充缺失值
        # fill missing values with mean
        mean_fill_features = ['bm', 'cfp', 'lev', 'agr', 'lgr', 'ps', 'quick', 'roaq', 'roeq', 'roic', 'sgr', 'tang']
        for feature in mean_fill_features:
            if feature not in self._df.columns:
                print(f"Feature '{feature}' not found in the DataFrame.")
                continue
            mean_value = self._df[feature].mean()
            self._df[feature].fillna(mean_value)

        # 下列特征变化平稳，可以采用线性插值法填充缺失值
        # fill missing values with linear interpolation
        linear_features = ['beta', 'bm_ia', 'chinv', 'currat', 'cashdebt', 'depr', 'dy', 'ep', 'gma', 'rd_sale']
        for feature in linear_features:
            if feature not in self._df.columns:
                print(f"Feature '{feature}' not found in the DataFrame.")
                continue
            self._df[feature] = self._df[feature].interpolate(method="linear")

        # 下列特征的变化与时间高度相关，采用时间序列插值法填充缺失值
        # fill missing values with time series interpolation
        time_series_features = ['mom1m', 'mom6m', 'std_turn', 'std_dolvol', 'turn', 'retvol', 'maxret', 'rsup']
        for feature in time_series_features:
            if feature not in self._df.columns:
                print(f"Feature '{feature}' not found in the DataFrame.")
                continue
            self._df[feature] = self._df[feature].interpolate(method="time")

        # if you just want to replace missing values with 0, comment the above codes

        # 其余全部缺失值用0填充
        # replace all missing values with 0
        self._df.fillna(0, inplace=True)

        # get missing values and print them again
        missing_values = self._df.isnull().sum()
        print("Missing values every column:\n", missing_values, "\n")

        # plot the feature distribution after processing
        if plt_feature in self._columns:
            plt.figure(figsize=(6, 4))
            plt.hist(self._df[plt_feature], bins=30, color="blue", alpha=0.7, label="Data (After Filling Missing Values)")
            plt.title(f"{plt_feature}: Distribution After Processing")
            plt.xlabel(plt_feature)
            plt.ylabel("Frequency")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.legend()
            plt.show()

    def _split_data(self):
        # 表格为时序数据，时序数据划分训练集、验证集和测试集必须是连续的，防止泄露未来信息
        # 下面划分比例跟Sample不一样
        # datas will be split into 60% training set, 20% validation set and 20% testing set
        n = len(self._df)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        train_data = self._df[:train_end]
        val_data = self._df[train_end:val_end]
        test_data = self._df[val_end:]

        print(f"Size of Training Set: {len(train_data)}")
        print(f"Size of Validation Set: {len(val_data)}")
        print(f"Size of Testing Set: {len(test_data)}")

        # get the target column
        target_column = "ret"

        self.dataset_X["train"], self.dataset_y["train"] = split_dataset_with_target(train_data, target_column)
        self.dataset_X["val"], self.dataset_y["val"] = split_dataset_with_target(val_data, target_column)
        self.dataset_X["test"], self.dataset_y["test"] = split_dataset_with_target(test_data, target_column)
   



def split_dataset_with_target(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y



def main():
    # 这是我的文件路径，在自己电脑跑记得改掉
    # for easier use, hard code in file path
    DATA_FILE_PATH = 'C:/Code/Python/FQA/finalwork/GHZ_ZHY_V8.csv'

    # create a StockDataProcessor object and preprocess the dataframe
    data_processor = StockDataProcessor(DATA_FILE_PATH)
    print(data_processor.get_df().info())

    data_processor.preprocess()


# main function
if __name__ == "__main__":
    main()
