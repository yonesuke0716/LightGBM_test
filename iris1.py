import lightgbm as lgb

from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np

"""LightGBM を使った多値分類のサンプルコード"""


def main():
    # Iris データセットを読み込む
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # 訓練データとテストデータに分割する
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # データセットを生成する
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # LightGBM のハイパーパラメータ
    lgbm_params = {
        # 多値分類問題
        "objective": "multiclass",
        # クラス数は 3
        "num_class": 3,
    }

    # 上記のパラメータでモデルを学習する
    model = lgb.train(lgbm_params, lgb_train, valid_sets=lgb_eval)

    # テストデータを予測する
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_max = np.argmax(y_pred, axis=1)  # 最尤と判断したクラスの値にする

    # 精度 (Accuracy) を計算する
    accuracy = sum(y_test == y_pred_max) / len(y_test)
    print(accuracy)


if __name__ == "__main__":
    main()
