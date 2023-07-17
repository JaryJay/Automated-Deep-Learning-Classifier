#from ScrawlerCore import ScrawlerCore

import threading
import time
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import urllib 
import os

output_folder = '../image/'

def GoogleImageScrawler(keyword, n = 400):
    #you need to call the Selenium Chrome driver, and the path to chromedriver.exe
    driver = webdriver.Chrome(executable_path='../chromedriver.exe')
    driver.get(f"https://www.google.com/search?q={keyword}&tbm=isch")

    image_links_count = 0
    last_height = driver.execute_script("return document.body.scrollHeight")
    #get enough image as you want
    while image_links_count <= n:
        image_links_count = len(driver.find_elements_by_class_name('rg_i.Q4LuWd'))
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        # scroll to bottom will automatically load until 100 images are found.
        if image_links_count <= n and new_height == last_height:
            try:
                element = driver.find_elements_by_class_name('mye4qd') 
                element[0].click()
            except:
                break
        last_height = new_height
    img_elements = driver.find_elements_by_class_name('rg_i.Q4LuWd')
    links = []
    #there are some images links might be None (since the collections of google images is also scrapped from different sources).
    for ele in img_elements:
        data_src = ele.get_attribute('data-src')
        if ele.get_attribute('data-src') is None:
            links.append(ele.get_attribute('src'))
        else:
            links.append(data_src)
        # Limit to n links
        if len(links) == n:
            break
    driver.quit()
    #excute the Selenium Chrome Driver, then print how many image did you get.
    print(f'{keyword}: Found {len(links)} links')

    if not os.path.isdir(output_folder + keyword):
        os.mkdir(output_folder + keyword)

    for i,link in enumerate(links):
        print(f'{keyword}: Downloading {i}/{len(links)}')
        name = f'{output_folder}{keyword}/google_img_{i}.png'
        urllib.request.urlretrieve(link, name)
        sleep(2)
    
    print(f'{keyword}: Finished downloading {len(links)} images')

keywords = ['Desk', 'Chair', 'Laptop']

starttime = time.time()

# Scrawl through all keywords in parallel
threads = [threading.Thread(target=GoogleImageScrawler, args=(k,)) for k in keywords]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

elapsed_time = time.time() - starttime
print(f'Finished in {elapsed_time:10.2f} seconds.')
