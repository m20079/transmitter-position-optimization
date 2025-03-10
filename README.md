# ディレクトリ構成

## transmitter_position_optimization

- main.py &rarr; main文
- print.py &rarr; コンソールに結果を出力
- save.py &rarr; ファイルに結果を出力
- graph.py &rarr; グラフやヒートマップとして出力
- constant.py &rarr; 定数の定義
- simulations.py &rarr; シミュレーションする関数を呼び出す

## transmitter_position_optimization/simulation

- single.py &rarr; 送信機が1つの場合のシミュレーションする関数
- double.py &rarr; 送信機が2つの場合のシミュレーションする関数
- triple.py &rarr; 送信機が3つの場合のシミュレーションする関数

## transmitter_position_optimization/environment

- coordinate.py &rarr; 座標に関わる関数など
- data_rate.py &rarr; 通信速度が最大の座標を取り出す and 座標から通信速度を取得する関数
- distance.py &rarr; 座標から距離を算出する関数
- evaluation.py &rarr; 最適値をどう定義するか(各受信機の平均 or 標準偏差 or 最小値)
- propagation.py &rarr; パスロス and シャドウイング and 通信路容量を計算する関数
- receivers.py &rarr; 受信機の座標や性能を保存するクラス

## transmitter_position_optimization/conventional_method

- distance_estimation.py &rarr; 受信機の位置から送信機の最適値を算出する関数
- random_search.py &rarr; ランダムサーチで送信機の最適値を算出する関数

## transmitter_position_optimization/bayesian_optimization

- acquisition.py &rarr; 獲得関数(ucb or pi or ei)
- bayesian_optimization.py &rarr; ベイズ最適化のプログラム
- gaussian_process_regression.py  &rarr; ガウス過程回帰のプログラム
- kernel &rarr; 様々なカーネル関数
- parameter_optimization &rarr; 様々なハイパーパラメータ算出プログラム
