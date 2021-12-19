import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from keras.models import  load_model

def predict(target_members):
    model = load_model('./my_model')
    cascade = cv2.CascadeClassifier("C:\\ProgramData\\Anaconda3\\Lib\site-packages\\cv2\data\\haarcascade_frontalface_alt.xml")

    PATH = "D:/NiziU/temp/"
    members = ['Mako', 'Rio', 'Maya', 'Riku', 'Ayaka', 'Mayuka', 'Rima', 'Miihi', 'Nina']
    print()

    for member in members:
        print(member)
        image=cv2.imread(os.path.join(PATH, member +".jpg"))
        if image is None:
            print("Not open:")
        b,g,r = cv2.split(image)
        image = cv2.merge([r,g,b])

        #opencvを使って顔抽出
        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
                name = target_members[nameNumLabel]
                cv2.putText(image,name,(x,y+height+20),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)

                dic = {}
                for i in range(len(target_members)):
                    dic[i] = predict[0][i]

                dic = sorted(dic.items(), reverse=True, key=lambda x : x[1])

                data = []
                for i in range(min(3, len(target_members))):
                    name = target_members[dic[i][0]]
                    percentage = round(dic[i][1] * 100, 1)

                    data.append([name, f'{percentage}%'])

                for d in data:
                    print(d)

        #顔が検出されなかった時
        else:
            print("no face")

        print()



def predict_image_in_directory():
    model = load_model('./my_model.h50')
    cascade = cv2.CascadeClassifier("C:\\ProgramData\\Anaconda3\\Lib\site-packages\\cv2\data\\haarcascade_frontalface_alt.xml")

    PATH = "D:/NiziU/temp2/*"

    import glob

    files = glob.glob(PATH)
    for img in files:
        image=cv2.imread(img)
        if image is None:
            print("Not open:")
        b,g,r = cv2.split(image)
        image = cv2.merge([r,g,b])

        #opencvを使って顔抽出
        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
                target_members = ['Mako', 'Rio', 'Maya', 'Riku', 'Ayaka', 'Mayuka', 'Rima', 'Miihi', 'Nina']
                name = target_members[nameNumLabel]
                cv2.putText(image,name,(x,y+height+20),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)

                dic = {}
                for i in range(len(target_members)):
                    dic[i] = predict[0][i]

                dic = sorted(dic.items(), reverse=True, key=lambda x : x[1])

                data = []
                for i in range(min(3, len(target_members))):
                    name = target_members[dic[i][0]]
                    percentage = round(dic[i][1] * 100, 1)

                    data.append([name, f'{percentage}%'])

                for d in data:
                    print(d)

        #顔が検出されなかった時
        else:
            print("no face")

        plt.imshow(image)
        plt.show()
        print()


if __name__ == '__main__':
    #predict(['Mako', 'Rio', 'Maya', 'Riku', 'Ayaka', 'Mayuka', 'Rima', 'Miihi', 'Nina'])
    predict_image_in_directory()
