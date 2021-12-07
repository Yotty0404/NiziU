import cv2
import glob
import os

IMAGE_FOLDER_PATH = 'D:\\NiziU\\Image_NiziU_Organize'
OUTPUT_FOLDER = "D:\\NiziU\\face"
members = ['Mako', 'Rio', 'Maya', 'Riku', 'Ayaka', 'Mayuka', 'Rima', 'Miihi', 'Nina']
members = ['Ayaka']
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for member in members:
    print(member)
    #元画像を取り出して顔部分を正方形で囲み、64×64pにリサイズ、別のファイルにどんどん入れてく
    files = os.listdir(os.path.join(IMAGE_FOLDER_PATH, member))
    os.makedirs(os.path.join(OUTPUT_FOLDER, member), exist_ok=True)
    for img in files:
        image=cv2.imread(os.path.join(IMAGE_FOLDER_PATH, member, img))
        if image is None:
            print("Not open:", img)
            continue

        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier("C:\\ProgramData\\Anaconda3\\Lib\site-packages\\cv2\data\\haarcascade_frontalface_alt.xml")
        # 顔認識の実行
        face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))
        #顔が１つ以上検出された時
        if len(face_list) > 0:
            for rect in face_list:
                x,y,width,height=rect
                image = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
                if image.shape[0]<64:
                    continue
                image = cv2.resize(image,(64,64))
                #保存
                fileName=os.path.join(OUTPUT_FOLDER, member, img)
                cv2.imwrite(str(fileName),image)
        #顔が検出されなかった時
        else:
            print(img + "：顔が検出できませんでした")
            continue

    print()