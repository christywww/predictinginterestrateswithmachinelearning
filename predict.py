from TextDataScraper import URLExtractor
from TextParagraphFeatureBuilder import ParagraphFeatureBuilder
from BondRateDataScraper import BondRateDataScraper
from Model import Model

# edit these parameters if you like!
model_type = "LSTM" # "SVM", "LSTM", or "MLP"
args = {"time_steps": 30, "batch_size": 4}
# input nasdaq data link API key
quandl_authtoken = "YOUR API KEY HERE"

urlextractor = URLExtractor()
urlextractor.process()
tp = ParagraphFeatureBuilder(urlextractor.text_data, topic_count=30)
tp.process() # test=True)
brds = BondRateDataScraper()
brds.process(textdf=tp.paragraph_df, authtoken=quandl_authtoken)

model = Model(modeltype=model_type, text_df=tp.feature_df, bondrate_df=brds.bond_rate_df, nr_topics=tp.topic_count, **args)
model.process()
print('')

tp.feature_df.to_csv("feature_df.csv")
brds.bond_rate_df.to_csv("bondrate_df.csv")

