import shutil
import os

PATH = 'D:/NiziU/'
files = os.listdir(PATH)
files_dir = [f for f in files if os.path.isdir(os.path.join(PATH, f))]
if 'face' in files_dir:
    shutil.rmtree(os.path.join(PATH, 'face'))
if 'train' in files_dir:
    shutil.rmtree(os.path.join(PATH, 'train'))
if 'test' in files_dir:
    shutil.rmtree(os.path.join(PATH, 'test'))

shutil.copytree(os.path.join(PATH, 'face_Organize'),os.path.join(PATH, 'face'))

from devide_test_train import devide_test_train
from inflation import inflation
from learn import learn
from predict import predict

#target_members = ['Mako', 'Rio', 'Maya', 'Riku', 'Ayaka', 'Mayuka', 'Rima', 'Miihi', 'Nina']
target_members = ['Mayuka','Maya']

print('devide_test_train開始')
devide_test_train()

print('inflation開始')
inflation()

print('learn開始')
learn(target_members)

print('predict開始')
predict(target_members)