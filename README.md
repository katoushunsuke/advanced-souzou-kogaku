# advanced-souzou-kogaku

## 概要
物体検出アルゴリズムyolo, 関節検知ライブラリmediapipe, 小型ドローンtelloを組み合わせたアプリケーションのコードを公開しています。各モジュールの詳細については以下のリンクを参照してください.
- [yolo v3](https://pjreddie.com/darknet/yolo/)
- [yolo v7](https://github.com/WongKinYiu/yolov7)
- [mediapipe](https://developers.google.com/mediapipe)
- [tello](https://www.ryzerobotics.com/jp/tello)

## 構成
ディレクトリ構成は以下の通りです．
- `mediapipe_div` : モデルの作成, 簡易動作試験のスクリプト
- `sliding` : 画像を分割してから物体を検出するスクリプト
- `tello-experiment` : telloを用いた実験用スクリプト. yolo v7を使用
- `yolov3-mediapipe` : 人検知処理と関節検知処理を組み合わせたスクリプト

## 初期設定
### YOLO v3
[weights](https://pjreddie.com/media/files/yolov2-tiny.weights)と[cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-tiny.cfg)を作業ディレクトリにダウンロードしてください．ファイルに任意の名前を付け, YOLO v3用スクリプトの`net = cv2.dnn.readNet("model_yolo.weights", "model_yolo.cfg")`の部分をファイル名に合わせて変更してください.
### YOLO v7
```
git clone https://github.com/WongKinYiu/yolov7
```
を pycharmのターミナルで実行してください．クローンが完了したら,
```
pip install requirements.txt
```
を実行して必要なモジュールをインストールします．
YOLO v3のときと同様に [weights](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt)をダウンロードして該当する箇所に変更を加えます. これによりdetect.pyの実行が可能になります.
### TELLO
telloを用いたデモコードを実行するためには, [MPlayer](https://drive.google.com/drive/folders/1qeqQ2OQrAgvcXlVddv23sgrmzCPKIIaB?usp=sharing)をPCにダウンロードし, PATHを通す必要があります.

## 内容
### Sliding
yolo v7は高速で高精度な検出を行うことができるアルゴリズムですが, 物体が密集している場合や物体が小さすぎる場合に検出漏れや誤検知が増加する傾向があります．以下の写真は検出漏れ・誤検知が増加する典型的な例です．
![cars](cars.jpg)
このような画像を直接ニューラルネットワークに通すのではなく，複数の画像に分割してから通すことは，検出漏れと誤検知を防ぐ効果的な手段です．下の画像は元の画像を16枚の画像に分割した後，それぞれの画像に対して検出を行った結果です．これにより，検出漏れが大幅に減少することが分かっています．[slide_detection.py](sliding/slide_detection.py)を実行することでデモンストレーションを行うことができます．
![cars_result](cars_result.png)