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

    url_lists = {
        # "boy": {
        #     "nub": "http://www.in-gender.com/Gender-Ultrasound/Gallery-Forum-Method.aspx?ID=2&M=Nub&G=M",
        #     "potty": "http://www.in-gender.com/Gender-Ultrasound/Gallery-Forum-Method.aspx?ID=1&M=Potty&G=M",
        #     "ramzi": "http://www.in-gender.com/Gender-Ultrasound/Gallery-Forum-Method.aspx?ID=3&M=Ramzi&G=M",
        #     "skull": "http://www.in-gender.com/Gender-Ultrasound/Gallery-Forum-Method.aspx?ID=4&M=Skull&G=M",
        #     "d3": "http://www.in-gender.com/Gender-Ultrasound/Gallery-Forum-Method.aspx?ID=6&M=3D&G=M",
        #     "other": "http://www.in-gender.com/Gender-Ultrasound/Gallery-Forum-Method.aspx?ID=7&M=Other&G=M",
        # },
        "girl": {
            "nub": "http://www.in-gender.com/Gender-Ultrasound/Gallery-Forum-Method.aspx?ID=2&M=Nub&G=F",
            "potty": "http://www.in-gender.com/Gender-Ultrasound/Gallery-Forum-Method.aspx?ID=1&M=Potty&G=F",
            "ramzi": "http://www.in-gender.com/Gender-Ultrasound/Gallery-Forum-Method.aspx?ID=3&M=Ramzi&G=F",
            "skull": "http://www.in-gender.com/Gender-Ultrasound/Gallery-Forum-Method.aspx?ID=4&M=Skull&G=F",
            "d3": "http://www.in-gender.com/Gender-Ultrasound/Gallery-Forum-Method.aspx?ID=6&M=3D&G=F",
            "other": "http://www.in-gender.com/Gender-Ultrasound/Gallery-Forum-Method.aspx?ID=7&M=Other&G=F",
        }
    }

    sleep_interval = 5
    data_dir = os.path.join(os.getcwd(), 'data')
    crawler = Driver()

    try:
        for gender in url_lists:
            for method in url_lists[gender].keys():

                index_url = url_lists[gender][method]
                page_id = 1

                # Selenium
                try:
                    driver = DriverConfig.driver
                    driver.get(index_url)

                    time.sleep(sleep_interval)
                    continue_checking = True

                    while (continue_checking):

                        # inject script
                        thread_urls = driver.execute_script(open("extract_index_urls.js").read())
                        with open('thread_urls.csv', 'a', newline='') as f:
                            writer = csv.writer(f)
                            if thread_urls:
                                for thread_url in thread_urls:
                                    writer.writerow([gender, method, page_id, index_url, thread_url])
                            else:
                                writer.writerow([gender, method, page_id, index_url, ""])

                        time.sleep(sleep_interval)

                        # see if there's a next
                        try:
                            element = driver.find_element_by_id("btnNext")
                            if element:
                                element.click()
                                page_id += 1
                                print("next {} {}".format(page_id, index_url))

                        # move onto next if not a next
                        except NoSuchElementException:
                            print("no next {} {}".format(page_id, index_url))
                            continue_checking = False

                except Exception as e:
                    print("Selenium error: {}".format(e))

    except Exception as e:
        print("Main error: {}".format(e))

    finally:
        driver.quit()

if __name__ == "__main__":
    main()
