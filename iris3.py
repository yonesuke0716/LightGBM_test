import lightgbm as lgb

import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""LightGBM を使った多値分類のサンプルコード"""


def main():
    # Iris データセットを読み込む
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # データセットを学習用とテスト用に分割する
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # テスト用のデータを評価用と検証用に分ける
    X_eval, X_valid, y_eval, y_valid = train_test_split(X_test, y_test, random_state=42)

    # データセットを生成する
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

    # LightGBM のハイパーパラメータ
    lgbm_params = {
        # 多値分類問題
        "objective": "multiclass",
        # クラス数は 3
        "num_class": 3,
    }

    # 上記のパラメータでモデルを学習する
    model = lgb.train(
        lgbm_params,
        lgb_train,
        # モデルの評価用データを渡す
        valid_sets=lgb_eval,
        # 最大で 1000 ラウンドまで学習する
        num_boost_round=1000,
        # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
        early_stopping_rounds=10,
    )

    # 学習したモデルでホールドアウト検証する
    y_pred_proba = model.predict(X_valid, num_iteration=model.best_iteration)
    # 返り値は確率になっているので最尤に寄せる
    y_pred = np.argmax(y_pred_proba, axis=1)

    # 精度 (Accuracy) を計算する
    accuracy = accuracy_score(y_valid, y_pred)
    print(accuracy)


if __name__ == "__main__":
    main()
