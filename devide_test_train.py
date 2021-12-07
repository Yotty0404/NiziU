import shutil
import random
import os

members = ['Mako', 'Rio', 'Maya', 'Riku', 'Ayaka', 'Mayuka', 'Rima', 'Miihi', 'Nina']

IMAGE_FOLDER_PATH = 'D:\\NiziU\\face'
OUTPUT_FOLDER = "D:\\NiziU\\test"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


for member in members:
    files = os.listdir(os.path.join(IMAGE_FOLDER_PATH, member))
    os.makedirs(os.path.join(OUTPUT_FOLDER, member), exist_ok=True)

    #img_file_name_listをシャッフル、そのうち2割をtest_imageディテクトリに入れる
    random.shuffle(files)
    # 2割をテストデータに移行
    for t in range(len(files)//5):
        shutil.move(os.path.join(IMAGE_FOLDER_PATH, member, files[t]), os.path.join(OUTPUT_FOLDER, member))