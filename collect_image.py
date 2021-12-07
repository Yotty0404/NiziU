import configparser
import httplib2
import json 
import os
import time
import urllib.request
from urllib.parse import quote
from icrawler.builtin import BingImageCrawler

from delete_same_image import delete_same_image

IMAGE_FOLDER_PATH = 'D:\\NiziU\\Image_NiziU'

members = ['Mako', 'Rio', 'Maya', 'Riku', 'Ayaka', 'Mayuka', 'Rima', 'Miihi', 'Nina']

for member in members:
    crawler = BingImageCrawler(storage={'root_dir': os.path.join(IMAGE_FOLDER_PATH, member)})
    crawler.crawl(keyword = f'NiziU {member}', max_num = 200)



#delete_same_image()


