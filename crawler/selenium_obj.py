import os
import selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service


class DriverConfig():
    """ Consistant driver config """

    options = webdriver.ChromeOptions()

    options.add_argument('--ignore-certificate-errors')
    options.add_argument("user-data-dir=C:\\Path") #Path to your chrome profile

    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option("excludeSwitches", ["enable-automation"])

    # chromdriver should match chrome browser version - https://chromedriver.chromium.org/downloads
    driver = webdriver.Chrome('crawler/chromedriver.exe', chrome_options=options)
    # driver.set_window_size(1920, 1080)


class Driver(DriverConfig):
    """ Main driver for navigating to webpage and interacting with it """

    def __init__(self):
        """ Initiate driver and load script """
        self.driver = DriverConfig.driver

    def quit(self, m=None):
        """ Quit the driver. Print a message if provided """
        print( "{}".format(m) ) if m else print( "quitting..." )
        self.driver.quit()

    def get_page(self, url, script):
        """ Try to extract data from url using script """

        status = 0
        extract = None

        try:
            self.driver.get(url)
            status = 1 # page loaded, but exact status (200, 404, ...) unknown
            extract = self.driver.execute_script(open(script).read())

        except Exception as e:
            print("selenium.get_page failed: {}".format(e))

        return status, extract

    def save_screenshot(self, full_path_name):
        self.driver.save_screenshot(full_path_name)

    def process_file(self, dir, filename, screenshot=None):
        """ Extract from local file """
        local_file = 'file://{}'.format(os.path.join(dir, filename))
        return self.get_page(local_file)



# Details of Firefox error:
# "firefox not loading beautybay."
# todo: look into it
# todo: Store both FF and Chrome driver configs in /configs

# class DriverConfig():
# from selenium.webdriver.firefox.options import Options
# from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
# from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
#     """ For Driver consistency, set things like
#     headless state, browser driver and window size """
#
#     ff_profile = webdriver.FirefoxProfile()
#     ff_profile.set_preference("general.useragent.override", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:80.0) Gecko/20100101 Firefox/80.0")
#
#     gecko = os.path.normpath(os.path.join(os.path.dirname(__file__), 'geckodriver'))
#     binary = FirefoxBinary(r'C:\Program Files\Firefox Developer Edition\firefox.exe')
#
#     options = Options()
#     options.headless=False
#
#     driver = webdriver.Firefox(
#         firefox_binary=binary,
#         executable_path=(gecko+'.exe'),
#         options=options,
#     )
#     driver.set_window_size(1920, 1080)
#     # driver.maximize_window()
