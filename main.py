"""
Copyright (c) 2022 株式会社 神戸デジタル・ラボ

This software is released under The MIT License.
https://licenses.opensource.jp/MIT/MIT.html
"""

'''
プログラム終了方法
 - qまたはEscを押下
'''

# 必要モジュールのインポート
from pickle import FALSE
import cv2  # OpenCV: カメラからの画像を読み取る
import requests  # requests: CustomVisionに最新スクリーンショットを送信し、その結果を受け取る
from time import time  # カメラからのスクリーンショット取得間隔を指定

###############################################
# ユーザー入力欄
###############################################
# Azure Custom Visionポータル -> Performance -> Prediction URL より取得可能
# ※ENDPOINTとKEYは外部に公開しないでください※
DET_ENDPOINT = "ENDPOINT_URL"
DET_KEY = "KEY"

CAMERA_NUM = 1   # 0:内蔵カメラ 1:USBカメラ

###############################################
# プログラム本体
###############################################
# Azure上にあるAIモデルにAPI経由でリクエストを送信する


def detection():
    url = DET_ENDPOINT
    headers = {'content-type': 'application/octet-stream',
               'Prediction-Key': DET_KEY}
    # tmpフォルダ保存された画像を送信
    response = requests.post(url, data=open(
        "./tmp/screen.jpg", "rb"), headers=headers)
    response.raise_for_status()
    # JSON形式で結果を受け取る
    analysis = response.json()
    # 検出した不良の数だけJSON形式で返ってくる
    if len(analysis["predictions"]) != 0:
        tag = analysis["predictions"][0]["tagName"]
        prob = analysis["predictions"][0]["probability"]
        box = analysis["predictions"][0]["boundingBox"]
        return tag, prob, box
    else:
        return "-", "-", {"left": 0, "top": 0, "width": 0, "height": 0}

###############################################
# OpenCVの設定
###############################################


# 読み込む対象のカメラの指定
cam = cv2.VideoCapture(CAMERA_NUM, cv2.CAP_DSHOW)
# Windowサイズの指定（横）
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# Windowサイズの指定（縦）
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# カメラ映像からのスクリーンショットを切り出すタイミングを指定(秒)
limit_time = 2
# 前回読み込んだ時間の初期化
previous_time = 0

###############################################
# メイン処理
###############################################
while True:
    try:
        # カメラ映像からスクリーンショットの取得
        _, img = cam.read()
        # 指定間隔を超えたら実行
        if limit_time < time() - previous_time:
            # スクリーンショットの取得
            cv2.imwrite('./tmp/screen.jpg', img)
            # CustomVisionに送信、タグ名と自信度をname, predに表示
            tag, prob, box = detection()
            # 入力画像のサイズ（高さ、幅）を取得
            height, width = img.shape[:2]

            # 前回取得時間を更新
            previous_time = time()

        # カメラ映像に表示する文字のフォントの設定
        font = cv2.FONT_HERSHEY_SIMPLEX

        """ 
        色によって一目で判別できるようにする
        白：良品
        黄: 要目視点検 (0.5 < prob =< 0.9)
        赤: 不良品（0.9 < prob）
        """

        text_color = (255, 255, 255)  # 白で初期化

        if prob > 0.5:
            text_color = (0, 255, 255)  # 黄
        if prob > 0.9:
            text_color = (0, 0, 255)  # 赤

        # バウンディングボックスの描画
        img = cv2.rectangle(img, (int(box["left"]*width), int(box["top"]*height)), (int(
            box["left"]*width+box["width"]*width), int(box["top"]*height+box["height"]*height)), text_color, 5)

        # タグ名の画面上の表示位置やサイズの設定
        cv2.putText(img, str(tag), (10, 100), font,
                    2, text_color, 3, cv2.LINE_AA)
        # 自信度の画面上の表示位置やサイズの設定
        cv2.putText(img, str(round(prob, 3)), (10, 180), font,
                    2, text_color, 3, cv2.LINE_AA)
        # 設定の反映と表示
        cv2.imshow("detect", img)

        # QかEscキー押下にてプログラムを終了できる設定
        if (cv2.waitKey(1) & 0xFF == ord("q")):
            break
    except:
        continue

# 終了キー押下後、後処理
cam.release()
cv2.destroyAllWindows()
