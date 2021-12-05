import configparser
import httplib2
import json 
import os
import time
import urllib.request
from urllib.parse import quote

from delete_same_image import delete_same_image

conf = configparser.ConfigParser()
conf.read('config.ini', encoding='utf-8')

API_KEY = conf['GoogleAPI']['KEY']
SEARCH_ENGINE_ID = conf['GoogleSearchEngine']['ID']

IMAGE_FOLDER_PATH = 'D:\Image_NiziU\\'

def get_extension(link):
    l_ext = ['.jpg', '.jpeg', '.gif', '.png', '.bmp']
    for ext in l_ext:
        index = link.find(ext)
        if index >= 0:
            return ext

    return -1


def get_image(member):
    for i in range(20):
        url = 'https://www.googleapis.com/customsearch/v1?key={}&cx={}&num={}&start={}&q={}&searchType=image'

        url = url.format(API_KEY, SEARCH_ENGINE_ID, 10 , i+1, quote(f'NiziU {member}'))

        res = urllib.request.urlopen(url)
        data = json.loads(res.read().decode('utf-8'))

        opener = urllib.request.build_opener()
        http = httplib2.Http('.cache')
        cnt = 0
        for d in data['items']:
            try:
                ext = get_extension(d['link'])

                if ext == -1:
                    continue

                response, content = http.request(d['link'])
                with open(os.path.join(IMAGE_FOLDER_PATH, member, f'{member}_{10*i + cnt}{ext}'), 'wb') as f:
                    f.write(content)

                cnt+=1

            except:
                print('failed to download images.')
                continue

        time.sleep(1)

members = ['Mako', 'Rio', 'Maya', 'Riku', 'Ayaka', 'Mayuka', 'Rima', 'Miihi', 'Nina']

#for i in range(6,9):
    #get_image(members[i])

delete_same_image()


