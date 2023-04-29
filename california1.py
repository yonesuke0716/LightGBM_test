import lightgbm as lgb
import pandas as pd

from sklearn import datasets
from sklearn.datasets import fetch_california_housing


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np

"""LightGBM を使った回帰のサンプルコード"""


def main():
    # california データセットを読み込む
    california_housing = fetch_california_housing()
    X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    y = pd.Series(california_housing.target)

    # 訓練データとテストデータに分割する
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # データセットを生成する
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # LightGBM のハイパーパラメータ
    lgbm_params = {
        # 回帰問題
        "objective": "regression",
        # RMSE (平均二乗誤差平方根) の最小化を目指す
        "metric": "rmse",
    }

    # 上記のパラメータでモデルを学習する
    model = lgb.train(lgbm_params, lgb_train, valid_sets=lgb_eval)

    # テストデータを予測する
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    # RMSE を計算する
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(rmse)


if __name__ == "__main__":
    main()
