import requests
import cv2
import numpy as np
import os
import urllib2
import json
from multiprocessing import Pool
key = "your key"
url = "https://maps.googleapis.com/maps/api/streetview?size=600x300&location={}&scale=2&fov=120&&heading={}&pitch={}&key={}"
cities = ['52.372278,4.892636', '41.015137,28.979530', '51.454514,-2.587910', '45.5017,73.5673', '-33.856159,151.21256', '50.850346,4.351721',
          '52.520007,13.404954', '31.230390,121.473702', '37.983810,23.727539', '31.629472,-7.981084', '41.385064,2.173403', '43.212161,2.353663'
          '43.710173,7.261953','45.464204,9.189982', '48.208174,16.373819']
def main():
    img_data = requests.get(url).content
    with open('image_name.jpg', 'wb+') as handler:
        handler.write(img_data)
        print("written")


def download_image(loc, heading, pitch,name):
    img_data = requests.get(url.format(loc, heading, pitch, key)).content
    with open('{}.jpg'.format(name), 'wb+') as handler:
        handler.write(img_data)
    return cv2.imread('{}.jpg'.format(name))

def download_pano(loc,name):
    headings = [0,90,180,270]
    pitches = [0, 45, -45]
    images = []
    for i in headings:
        for j in pitches:
            im = download_image(loc, i, j,name)
            images.append(im)
    stitcher = cv2.createStitcher(True)
    result = stitcher.stitch(images)
    if result[0] is not 0:
        return
    mask = result[1] == 0
    mask = mask[:,:,0] + mask[:,:,1] + mask[:,:,2]
    mask = mask.astype(np.uint8)
    im = cv2.inpaint(result[1],mask,3,cv2.INPAINT_TELEA)
    path = '/home/gavin/scene_data/'
    cv2.imwrite('{}/google/{}.jpg'.format(path, name), im)
    #os.system('python3 expand.py {}/google/{}.jpg --out {}/predictions/'.format(path,name,path))

def get_city((city,id)):
    req = urllib2.Request(places_url.format(city))
    opener = urllib2.build_opener()
    f = opener.open(req)
    jsondata = json.loads(f.read())
    for place in jsondata['results']:
        lat = place['geometry']['location']['lat']
        lng = place['geometry']['location']['lng']
        places.append("{},{}".format(lat, lng))
    for i, loc in enumerate(places):
        download_pano(loc, id+i)

if __name__ == "__main__":
    u_a = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.82 Safari/537.36"
    places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={}&radius=20000&key={}"
    places = []
    args = []
    for i,city in enumerate(cities):
        args.append((city, i*20))
    p = Pool(8)
    p.map(get_city, args)

