import glob
import os

def delete_same_image():
    members = ['Mako', 'Rio', 'Maya', 'Riku', 'Ayaka', 'Mayuka', 'Rima', 'Miihi', 'Nina']
    IMAGE_FOLDER_PATH = 'D:\Image_NiziU\\'

    for member in members:
        files = glob.glob(os.path.join(IMAGE_FOLDER_PATH, member, '*'))
        dic = {}
        for file in files:
            dic[file] = os.path.getsize(file)

        # 削除するファイルを溜めていく
        s_delete = set()

        for k in dic:
            search_v = dic[k]
            keys = [k for k, v in dic.items() if v == search_v]

            # 1つは残して、他を削除するものを追加
            s_delete.update(keys[1:])

        for file in s_delete:
            os.remove(file)