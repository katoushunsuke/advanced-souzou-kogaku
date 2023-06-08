# advanced-souzou-kogaku

## Abstract
物体検出アルゴリズム'yolo', 関節検知ライブラリ'mediapipe', 小型ドローン'tello'を組み合わせたアプリケーションのコードを公開しています。
- [yolo v3](https://pjreddie.com/darknet/yolo/)
- [yolo v7](https://github.com/WongKinYiu/yolov7)
- [mediapipe](https://developers.google.com/mediapipe)
- [tello](https://www.ryzerobotics.com/jp/tello)

## Composition
ディレクトリ構成は以下の通りです．
- mediapipe_div : オリジナルデータからのモデルの作成, 簡易動作試験のスクリプト
- sliding : 画像を分割してから物体を検出するスクリプト
- tello-experiment : telloを用いた実験用スクリプト
- yolov3-mediapipe : 人検知処理と関節検知処理を組み合わせたスクリプト

## 3Dmodels
EAGLEにデフォルトで3Dデータが登録されていない部品の3Dデータ(*step)をアップロードしてください。
分類方法はlibrariesと同様です。
## documets
参考資料およびシステム図
物体検出アルゴリズム'yolo'を用いたアプリケーションです．
