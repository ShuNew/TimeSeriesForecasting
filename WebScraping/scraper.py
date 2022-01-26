import cloudscraper
import pandas as pd
from googlesearch import search

def dow30_components():
    scraper = cloudscraper.create_scraper(browser='chrome')
    url = "https://www.slickcharts.com/dowjones"
    source = scraper.get(url).text
    html_tables = pd.read_html(source)
    df = html_tables[0]
    return df

def sp500_components():
    scraper = cloudscraper.create_scraper(browser='chrome')
    url = "https://www.slickcharts.com/sp500"
    source = scraper.get(url).text
    html_tables = pd.read_html(source)
    df = html_tables[0]
    return df

def table_scraper(url, table_number):
    scraper = cloudscraper.create_scraper(browser='chrome')
    source = scraper.get(url).text
    html_tables = pd.read_html(source)
    df = html_tables[table_number]
    return df

def Gsearch(query, num_of_results):
    result_list = []
    for j in search(query, tld="com", num=num_of_results, stop=num_of_results, pause=1):
        result_list.append(j)
    return result_list
