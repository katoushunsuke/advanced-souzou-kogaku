# advanced-souzou-kogaku

## 概要
物体検出アルゴリズムyolo, 関節検知ライブラリmediapipe, 小型ドローンtelloを組み合わせたアプリケーションのコードを公開しています。
- [yolo v3](https://pjreddie.com/darknet/yolo/)
- [yolo v7](https://github.com/WongKinYiu/yolov7)
- [mediapipe](https://developers.google.com/mediapipe)
- [tello](https://www.ryzerobotics.com/jp/tello)

## 構成
ディレクトリ構成は以下の通りです．
- mediapipe_div : モデルの作成, 簡易動作試験のスクリプト
- sliding : 画像を分割してから物体を検出するスクリプト
- tello-experiment : telloを用いた実験用スクリプト. yolo v7を使用
- yolov3-mediapipe : 人検知処理と関節検知処理を組み合わせたスクリプト

## 初期設定
# YOLO v3
[weights](https://pjreddie.com/media/files/yolov2-tiny.weights)と[cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-tiny.cfg)を作業ディレクトリにダウンロードしてください．
`net = cv2.dnn.readNet("model_yolo.weights", "model_yolo.cfg")`
## documets
参考資料およびシステム図
物体検出アルゴリズム'yolo'を用いたアプリケーションです．
