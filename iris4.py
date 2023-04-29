import lightgbm as lgb

from sklearn import datasets
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

"""LightGBM を使った特徴量の重要度の可視化"""


def main():
    # Iris データセットを読み込む
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # 訓練データとテストデータに分割する
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # データセットを生成する
    lgb_train = lgb.Dataset(X_train, y_train, feature_name=iris.feature_names)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # LightGBM のハイパーパラメータ
    lgbm_params = {
        # 多値分類問題
        "objective": "multiclass",
        # クラス数は 3
        "num_class": 3,
    }

    # 上記のパラメータでモデルを学習する
    model = lgb.train(lgbm_params, lgb_train, valid_sets=lgb_eval, num_boost_round=40)

    # 特徴量の重要度をプロットする
    lgb.plot_importance(model, figsize=(12, 6))
    plt.show()


if __name__ == "__main__":
    main()
