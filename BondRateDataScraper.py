import nasdaqdatalink as quandl
import pandas as pd
from datetime import datetime

class BondRateDataScraper:
    def process(self, textdf_path=None, textdf=None, save_path=None, authtoken=None):
        if authtoken == None:
            print('An API key for the NASDAQ data link is needed. Please get one here: https://docs.data.nasdaq.com/v1.0/docs/getting-started')
            return

        print(datetime.now(), "Scraping bond rate data from NASDAQ Data Link...")
        data = quandl.get("USTREASURY/YIELD", authtoken=authtoken)
        data['date'] = data.index.values
        data['date'] = pd.to_datetime(data['date']).dt.date
        data = data[['date', '3 MO']]

        if textdf is None and textdf_path is None:
            print("Must either input the paragraphdf (as textdf) or path to paragraphdf's csv (as textdf_path)!")
            return
        if textdf_path is not None:
            minutesdata = pd.read_csv(textdf_path)
        if textdf is not None:
            minutesdata = textdf

        minutesdata['date'] = pd.to_datetime(minutesdata['date']).dt.date
        meetingdates = set(minutesdata.date.values)
        data['15d change'] = data['3 MO'].diff(periods=-15) * -1
        data['30d change'] = data['3 MO'].diff(periods=-30) * -1

        data = data[data['date'].isin(meetingdates)]
        self.bond_rate_df = data
        self.bond_rate_df.set_index('date', inplace=True)

        if save_path is not None:
            self.bond_rate_df.to_csv(save_path, index=False)
            print(datetime.now(), "saved bond rate data as", save_path)


if __name__ == "__main__":
    brds = BondRateDataScraper()
    brds.process(textdf_path="paragraph_df.csv", save_path="bondrate_df.csv")