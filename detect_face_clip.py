import cv2
import glob
import keyboard
import os
import time
import numpy as np

from PIL import ImageGrab, Image


OUTPUT_FOLDER = "D:\\NiziU\\face_clip"

cnt = 601


while True:
    if keyboard.read_key() == 'f10':
        
        im = ImageGrab.grabclipboard()
        if isinstance(im, Image.Image):
            print('has image')
        else:
            print('no image')
            continue

        # クリップボードの画像から顔部分を正方形で囲み、64×64にリサイズ、別のファイルにどんどん入れてく
        image = np.asarray(im)
        if image is None:
            print("Not open:", img)
            continue

        #BGRからRGBへ変換
        image = image[:, :, ::-1].copy()

        cascade = cv2.CascadeClassifier("C:\\ProgramData\\Anaconda3\\Lib\site-packages\\cv2\data\\haarcascade_frontalface_alt.xml")
        # 顔認識の実行
        face_list = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))
        # 顔が１つ以上検出された時
        if len(face_list) > 0:
            for rect in face_list:
                x,y,width,height = rect
                faceimage = image[rect[1]:rect[1] + rect[3],rect[0]:rect[0] + rect[2]]
                if faceimage.shape[0] < 64:
                    continue
                faceimage = cv2.resize(faceimage,(64,64))
                #保存
                fileName = os.path.join(OUTPUT_FOLDER, str(cnt) + '.jpg')
                cv2.imwrite(str(fileName),faceimage)

                cnt+=1

        # 顔が検出されなかった時
        else:
            print("顔が検出できませんでした")

        time.sleep(0.5)

