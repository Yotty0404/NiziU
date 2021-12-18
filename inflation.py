import os
import cv2
from scipy import ndimage
"""
faceディレクトリから画像を読み込んで回転、ぼかし、閾値処理をしてtrainディレクトリに保存する.
"""

def inflation():
    members = ['Mako', 'Rio', 'Maya', 'Riku', 'Ayaka', 'Mayuka', 'Rima', 'Miihi', 'Nina']

    IMAGE_FOLDER_PATH = 'D:\\NiziU\\face'
    OUTPUT_FOLDER = "D:\\NiziU\\train"

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for member in members:
        images = os.listdir(os.path.join(IMAGE_FOLDER_PATH, member))
        os.makedirs(os.path.join(OUTPUT_FOLDER, member), exist_ok=True)

        for i in range(len(images)):
            img = cv2.imread(os.path.join(IMAGE_FOLDER_PATH, member, images[i]))
            # 回転
            for ang in [-10,0,10]:
                img_rot = ndimage.rotate(img,ang)
                img_rot = cv2.resize(img_rot,(64,64))
                fileName=os.path.join(OUTPUT_FOLDER, member, str(i)+"_"+str(ang)+".jpg")
                cv2.imwrite(str(fileName),img_rot)
                # 閾値
                img_thr = cv2.threshold(img_rot, 100, 255, cv2.THRESH_TOZERO)[1]
                fileName=os.path.join(OUTPUT_FOLDER, member, str(i)+"_"+str(ang)+"_thr.jpg")
                cv2.imwrite(str(fileName),img_thr)
                # ぼかし
                img_filter = cv2.GaussianBlur(img_rot, (5, 5), 0)
                fileName=os.path.join(OUTPUT_FOLDER, member, str(i)+"_"+str(ang)+"_filter.jpg")
                cv2.imwrite(str(fileName),img_filter)



if __name__ == '__main__':
    inflation()