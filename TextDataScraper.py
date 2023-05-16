import requests
import re
from bs4 import BeautifulSoup
from datetime import datetime

class URLExtractor:
    def __init__(self):
        self.text_data = {}

    def get_urls(self):
        minutes_links = []

        # get list of urls from fomc_historical_year
        for year in range(1993, 2017):
            html = requests.get(url=f"""https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm""").text

            if year < 1996:
                searchstring = '"\/fomc\/MINUTES\/[0-9]+\/[0-9]+min.htm"'
            elif year < 2008:
                searchstring = '"\/fomc\/minutes\/[0-9]+.htm"'
            else:
                searchstring = '"\/monetarypolicy\/fomcminutes[0-9]+.htm"'

            links_in_year = re.findall(searchstring, html, re.IGNORECASE)
            if len(links_in_year) != 8: # double-check to see if all years have 8 meeting minutes
                print(f"""Year {year} only has {len(links_in_year)} links: {[''.join([char if char.isdigit() else '' for char in url]) for url in links_in_year]}""")

            minutes_links += re.findall(searchstring, html, re.IGNORECASE)

        # three of them weren't working for some reason, adding them back in manually
        minutes_links += ['"/monetarypolicy/fomcminutes20071031.htm"', '"/monetarypolicy/fomcminutes20071211.htm"', '"/monetarypolicy/fomc20080625.htm"']

        # get list of urls for 2017-22 meeting minutes on fomccalendars
        html = requests.get(url = 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm').text
        minutes_links += re.findall('"\/monetarypolicy\/fomcminutes[0-9]+.htm"', html)

        self.minutes_links = minutes_links

    def get_text(self, url):
        def process_filter_paragraph(para):
            nonlocal minutes_text #, curr_header

            # for boldedtitle in para.find_all('strong'):
                # curr_header = boldedtitle.get_text()

            text = para.get_text()

            text = text.replace('\n', ' ')
            text = text.replace('\r', ' ')
            text = text.replace('	', ' ')
            text = re.sub(' +', ' ', text)
            text = text.strip()

            # filter away short paragraphs, bullet point paragraphs, and paragraphs not ending with a full stop
            if len(text) > 500 and not (re.match('^ ?(\()?[0-9]?[a-z]?[iv]*\)', text, re.IGNORECASE) or re.match('^ ?[0-9]?[a-z]?[iv]*\.', text, re.IGNORECASE)) and text.endswith(".") and not 'by unanimous vote' in text.lower():
                if text.startswith("In conducting operations pursuant to the authorization and direction of the Federal Open Market Committee"): return
                if len(text.split()) > 400:
                    # split paragraphs with over 400 words into two to not cause trouble for vectorizer later
                    halflen = int(len(text.split('.')) / 2)
                    minutes_text.append(' '.join(text.split('.')[:halflen]))
                    minutes_text.append(' '.join(text.split('.')[halflen:]))
                else:
                    minutes_text.append(text)

        minutes_text = []

        link = 'https://www.federalreserve.gov/' + url[1:-1]
        html = requests.get(url=link).text
        html = re.sub('<strong>.*<\/strong>', '', html, flags=re.IGNORECASE) # remove titles (bolded words)

        parsed_html = BeautifulSoup(html, 'html.parser')

        # curr_header = ""

        for para in parsed_html.find_all("p"):

            if re.search('<\/p>(<\/p>)+', str(para)):
                # some documents use <p> with closing tags all at the bottom, which show up as one large chunk here, need to split them
                para = str(para).replace('</p>', '')
                splitted_paras = str(para).split('<p>')
                for splitted_para in splitted_paras:
                    process_filter_paragraph(BeautifulSoup(splitted_para, 'html.parser'))
                if len(str(para)) > 10000:
                    break
            else:
                process_filter_paragraph(para)

        if len(minutes_text) == 0:
            print(f"""Cannot extract text for {''.join([char for char in url if char in '0123456789'])} :(""")

        self.text_data[''.join([char for char in url if char in '0123456789'])] = minutes_text

    def save_text(self, filepath):
        import pickle
        file = open(filepath, 'wb')
        pickle.dump(self.text_data, file)
        file.close()
        print(datetime.now(), 'Saved textdata dictionary at', filepath)

    def process(self):
        print(datetime.now(), 'Getting URLs...')
        self.get_urls()
        print(datetime.now(), 'Extracting minutes text...')
        for url in self.minutes_links:
            self.get_text(url)

        self.save_text('textdata.pkl')
        print('')


if __name__ == "__main__":
    URLExtractor = URLExtractor()
    URLExtractor.get_text('"/fomc/minutes/19960326.htm"') #'"monetarypolicy/fomcminutes20220126.htm"')
    # print('')
    # URLExtractor.process()