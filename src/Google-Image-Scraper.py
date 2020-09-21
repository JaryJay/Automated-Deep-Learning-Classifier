#from ScrawlerCore import ScrawlerCore

from time import sleep
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import urllib 
import os

def GoogleImageScrawler(keyword, n = 400):
    #you need to call the Selenium Chrome driver, and the path to chromedriver.exe
    driver = webdriver.Chrome(executable_path='../chromedriver.exe')
    driver.get("https://www.google.com/search?q={}&tbm=isch".format(keyword))

    imageLinksCount = 0
    last_height = driver.execute_script("return document.body.scrollHeight")
    #get enough image as you want
    while imageLinksCount <= n:
        imageLinksCount = len(driver.find_elements_by_class_name('rg_i.Q4LuWd'))
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        # scroll to bottom will automatically load until 100 images are found.
        if imageLinksCount <= n and new_height == last_height:
            try:
                element = driver.find_elements_by_class_name('mye4qd') 
                element[0].click()
            except:
                break
        last_height = new_height
    imgElements = driver.find_elements_by_class_name('rg_i.Q4LuWd')
    links = []
    #there are some images links might be None (since the collections of google images is also scrapped from different sources).
    for ele in imgElements:
        data_src = ele.get_attribute('data-src')
        if ele.get_attribute('data-src') is None:
            links.append(ele.get_attribute('src'))
        else:
            links.append(data_src)
    driver.quit()
    #excute the Selenium Chrome Driver, then print how many image did you get.
    print(len(links))

    for i,link in enumerate(links):
        print('Downloading {}/{}'.format(i, len(links)))
        name = '../image/'+ string_name + '/' +'google_img_{}.png'.format(i)
        urllib.request.urlretrieve(link, name)
        sleep(2)

#use GoogleImageScrawler function with two parameters
#first parameter with a string as what's the name of the Image
#second parameter with the num of image you want
GoogleImageScrawler('Cat', 10)
string_name = 'Cat'
path = "please input the image folder path here"
os.mkdir(path, string_name)
