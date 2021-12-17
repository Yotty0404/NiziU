import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import  load_model
import sys

def detect_face(image):
    model = load_model('./my_model.h50')
    image=cv2.imread("C:/Users/tyuke/Desktop/temp/000041.jpg")
    if image is None:
        print("Not open:")
    b,g,r = cv2.split(image)
    image = cv2.merge([r,g,b])

    #print(image.shape)
    #opencvを使って顔抽出
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier("C:\\ProgramData\\Anaconda3\\Lib\site-packages\\cv2\data\\haarcascade_frontalface_alt.xml")

    # 顔認識の実行
    face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))
    #顔が１つ以上検出された時
    if len(face_list) > 0:
        for rect in face_list:
            x,y,width,height=rect
            cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (255, 0, 0), thickness=3)
            img = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            if image.shape[0]<64:
                print("too small")
                continue
            img = cv2.resize(image,(64,64))
            img=np.expand_dims(img,axis=0)

            name=""
            predict = model.predict(img)
            nameNumLabel=np.argmax(model.predict(img))
            members = ['Mako', 'Rio', 'Maya', 'Riku', 'Ayaka', 'Mayuka', 'Rima', 'Miihi', 'Nina']
            #members = ['Mako', 'Rio', 'Ayaka']
            name = members[nameNumLabel]
            cv2.putText(image,name,(x,y+height+20),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)

            print(predict)

            

            dic = {}
            for i in range(len(members)):
                dic[i] = predict[0][i]

            dic = sorted(dic.items(), reverse=True, key=lambda x : x[1])
            print()

            data = []
            for i in range(3):
                name = members[dic[i][0]]
                percentage = round(dic[i][1] * 100, 1)

                data.append([name, f'{percentage}%'])

            for d in data:
                print(d)

    #顔が検出されなかった時
    else:
        print("no face")
    return image


if __name__ == '__main__':
    whoImage=detect_face("C:/Users/tyuke/Desktop/maya.jpg")

    plt.imshow(whoImage)
    plt.show()