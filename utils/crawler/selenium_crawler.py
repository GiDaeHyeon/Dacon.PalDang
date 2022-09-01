import os
import csv
import time

from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from retry import retry


class SeleniumCrawler(object):
    def __init__(self, driver_dir: str = './chromedriver') -> None:
        self.driver = Chrome(executable_path=driver_dir)

    @staticmethod
    def set_default_by(element: dict, **kwargs) -> dict:
        by = element.get('by', None)
        if by is None:
            element['by'] = By.XPATH
        return element

    @retry(exceptions=(Exception,), tries=5, delay=5, max_delay=10)
    def get(self, url: str, **kwargs) -> bool:
        try:
            self.driver.get(url)
        except Exception as E:
            print(E)
            return False
        else:
            return True

    @retry(exceptions=(StaleElementReferenceException, ), tries=5, delay=5, max_delay=10)
    def get_value(self, element: dict, **kwargs) -> str:
        element = self.set_default_by(element=element)
        return self.driver.find_element(**element).text

    def send_key(self, element: dict, **kwargs) -> bool:
        element = self.set_default_by(element=element)
        web_element = self.driver.find_element(**element)
        try:
            web_element.clear()
            web_element.send_keys(element.get('key'))
            time.sleep(.1)
        except KeyError as E:
            print(E)
            raise ValueError
        except Exception as E:
            print(E)
            return False
        else:
            return True

    def click(self, element: dict, **kwargs) -> bool:
        element = self.set_default_by(element=element)
        web_element = self.driver.find_element(**element)
        try:
            web_element.click()
        except Exception as E:
            print(E)
            return False
        else:
            return True


if __name__ == '__main__':
    import pandas as pd
    from tqdm import tqdm

    target_time = pd.read_csv('../../competition_data/norm/test_y.csv', encoding='utf-8')['ymdhm'].tolist()
    target_xpaths = {'wl_1018662': '//*[@id="contents"]/table/tbody/tr[47]/td[3]',  # 청담대교
                     'wl_1018680': '//*[@id="contents"]/table/tbody/tr[44]/td[3]',  # 잠수교,
                     'wl_1018683': '//*[@id="contents"]/table/tbody/tr[48]/td[3]',  # 한강대교
                     'wl_1019630': '//*[@id="contents"]/table/tbody/tr[49]/td[3]',  # 행주대교
                     }
    input_box = '//*[@id="datetime"]'

    crawler = SeleniumCrawler()
    crawler.get('http://www.hrfco.go.kr/sumun/waterlevelList.do')
    time.sleep(5)

    try:
        with open(f'./data/target.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ymdhm'] + list(target_xpaths.keys()))
    except FileNotFoundError:
        os.makedirs('./data')
        with open(f'./data/target.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ymdhm'] + list(target_xpaths.keys()))

    record = None
    for idx, t in tqdm(enumerate(target_time), total=len(target_time)):
        before_record = record
        record = [t]
        crawler.send_key(element={'value': input_box, 'key': t})
        crawler.click(element={'value': '//*[@id="contents"]/div[3]/button'})
        time.sleep(1)

        if idx == 0:
            for v in target_xpaths.values():
                record.append(float(crawler.get_value({'value': v})))
        else:
            if before_record == record:
                time.sleep(5)
            for v in target_xpaths.values():
                record.append(float(crawler.get_value({'value': v})))

        with open(f'./data/target.csv', 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(record)
