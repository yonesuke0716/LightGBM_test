import lightgbm as lgb

from sklearn import datasets

import numpy as np

from matplotlib import pyplot as plt

"""LightGBM を使った多値分類のサンプルコード (CV)"""


def main():
    # Iris データセットを読み込む
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # データセットを生成する
    lgb_train = lgb.Dataset(X, y)

    # LightGBM のハイパーパラメータ
    lgbm_params = {
        # 多値分類問題
        "objective": "multiclass",
        # クラス数は 3
        "num_class": 3,
    }

    # 上記のパラメータでモデルを学習〜交差検証までする
    cv_results = lgb.cv(lgbm_params, lgb_train, nfold=10)
    cv_logloss = cv_results["multi_logloss-mean"]
    round_n = np.arange(len(cv_logloss))

    plt.xlabel("round")
    plt.ylabel("logloss")
    plt.plot(round_n, cv_logloss)
    plt.show()


if __name__ == "__main__":
    main()
