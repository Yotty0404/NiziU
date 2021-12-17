import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

members = ['Mako', 'Rio', 'Maya', 'Riku', 'Ayaka', 'Mayuka', 'Rima', 'Miihi', 'Nina']
#members = ['Mako', 'Rio', 'Ayaka']

TRAIN_FOLDER_PATH = 'D:\\NiziU\\train'
TEST_FOLDER_PATH = 'D:\\NiziU\\test'

# 教師データのラベル付け
X_train = [] 
Y_train = [] 
for i in range(len(members)):
    images = os.listdir(os.path.join(TRAIN_FOLDER_PATH, members[i]))
    for image in images:
        img = cv2.imread(os.path.join(TRAIN_FOLDER_PATH, members[i], image))
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        X_train.append(img)
        Y_train.append(i)

# テストデータのラベル付け
X_test = [] # 画像データ読み込み
Y_test = [] # ラベル（名前）
for i in range(len(members)):
    images = os.listdir(os.path.join(TEST_FOLDER_PATH, members[i]))
    for image in images:
        img = cv2.imread(os.path.join(TEST_FOLDER_PATH, members[i], image))
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        X_test.append(img)
        Y_test.append(i)
X_train=np.array(X_train)
X_test=np.array(X_test)

from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Dropout
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from tensorflow.keras import optimizers

y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)

# モデルの定義
#model = Sequential()
#model.add(Conv2D(input_shape=(64, 64, 3), filters=32,kernel_size=(3, 3), 
#                 strides=(1, 1), padding="same"))

## 追加
#model.add(Conv2D(filters=32, kernel_size=(3, 3), 
#                 strides=(1, 1), padding="same"))
#model.add(BatchNormalization())


#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(filters=32, kernel_size=(3, 3), 
#                 strides=(1, 1), padding="same"))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(filters=32, kernel_size=(3, 3), 
#                 strides=(1, 1), padding="same"))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Flatten())
#model.add(Dense(256))
#model.add(Activation("sigmoid"))
#model.add(Dense(128))
#model.add(Activation('sigmoid'))
#model.add(Dense(9))
#model.add(Activation('softmax'))

## コンパイル
#model.compile(optimizer='sgd',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',
                 input_shape=(64, 64, 3), padding="same"))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(9, activation='softmax'))
#model.add(Dense(3, activation='softmax'))


model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])

#model.compile(optimizer='sgd',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])

# 学習
history = model.fit(X_train, y_train, batch_size=32, 
                    epochs=60, verbose=1, validation_data=(X_test, y_test))

# 汎化制度の評価・表示
score = model.evaluate(X_test, y_test, batch_size=32, verbose=0)
print('validation loss:{0[0]}\nvalidation accuracy:{0[1]}'.format(score))

#acc, val_accのプロット
plt.plot(history.history["accuracy"], label="acc", ls="-", marker="o")
plt.plot(history.history["val_accuracy"], label="val_acc", ls="-", marker="x")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()

#モデルを保存
model.save("my_model.h50")