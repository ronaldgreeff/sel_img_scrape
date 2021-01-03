import os
import csv
import time
from urllib.parse import urlparse
from urllib.request import urlretrieve
from crawler.selenium_obj import Driver
import selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
# from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from crawler.selenium_obj import DriverConfig

def main():

    sleep_interval = 5
    data_dir = os.path.join(os.getcwd(), 'data')
    crawler = Driver()

    csv_files = ['thread_urls__boys.csv', 'thread_urls__girls.csv']

    try:
        driver = DriverConfig.driver

        for csv_file in csv_files:

            clean_list = []
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:

                    thread_url = row[-1]

                    if thread_url not in clean_list:
                        clean_list.append(thread_url)

                        gender = row[0]

                        gender_dir = os.path.join(data_dir, gender)
                        if not os.path.exists(gender_dir):
                            os.makedirs(gender_dir)

                        method = row[1]

                        method_dir = os.path.join(gender_dir, method)
                        if not os.path.exists(method_dir):
                            os.makedirs(method_dir)

                        driver.get(thread_url)
                        image_urls = driver.execute_script(open("extract_image_urls.js").read())

                        thread_id = urlparse(thread_url).query.split('=')[1]

                        for image_url in image_urls:
                            url_img_name = "".join(urlparse(image_url).path.split('/')[-1])
                            image_name = "{}_{}".format(thread_id, url_img_name)

                            try:
                                urlretrieve(image_url, os.path.join(method_dir, image_name))
                                print("got image {}".format(image_name))
                                time.sleep(sleep_interval)

                            except Exception as e:
                                print(e)
    finally:
        driver.quit()

    # try:
    #     for gender in url_lists:
    #         for method in url_lists[gender].keys():
    #
    #             index_url = url_lists[gender][method]
    #             page_id = 1
    #
    #             # Selenium
    #             try:
    #                 driver = DriverConfig.driver
    #                 driver.get(index_url)
    #
    #                 time.sleep(sleep_interval)
    #                 continue_checking = True
    #
    #                 while (continue_checking):
    #
    #                     # inject script
    #                     thread_urls = driver.execute_script(open("extract_index_urls.js").read())
    #                     with open('thread_urls.csv', 'a', newline='') as f:
    #                         writer = csv.writer(f)
    #                         if thread_urls:
    #                             for thread_url in thread_urls:
    #                                 writer.writerow([gender, method, page_id, index_url, thread_url])
    #                         else:
    #                             writer.writerow([gender, method, page_id, index_url, ""])
    #
    #                     time.sleep(sleep_interval)
    #
    #                     # see if there's a next
    #                     try:
    #                         element = driver.find_element_by_id("btnNext")
    #                         if element:
    #                             element.click()
    #                             page_id += 1
    #                             print("next {} {}".format(page_id, index_url))
    #
    #                     # move onto next if not a next
    #                     except NoSuchElementException:
    #                         print("no next {} {}".format(page_id, index_url))
    #                         continue_checking = False
    #
    #             except Exception as e:
    #                 print("Selenium error: {}".format(e))
    #
    # except Exception as e:
    #     print("Main error: {}".format(e))
    #
    # finally:
    #     driver.quit()

if __name__ == "__main__":
    main()
