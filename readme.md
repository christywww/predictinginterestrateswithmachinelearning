## Predicting Interest Rates with FOMC Meeting Minutes

### Project Overview
Interest rates play an important role in both the economy and financial markets. As such, there are many parties that would benefit from being able to predict changes in interest rates ahead of time. This project aims to extract features from meeting minutes of the Federal Open Market Committee (FOMC), one of the two committees in charge of monetary policy in the US, to generate predictions for changes in interest rate with the use of machine learning models using features generated from applying the BERT-based models FinBERT and BERTopic. Three models (using the LSTM, SVM and MLP architectures respectively) were trained and were evaluated based on their Mean Average Error (MAE) and Mean Squared Error (MSE). Of the three models, the LSTM model performed the best, and was able to predict the directional change in bond rate for 64.5% of the test set.

The full report for this project can be found [here](https://drive.google.com/file/d/1lHt0G1qiHGQzFnfBRYaxCCGWFzypZDT6/view?usp=sharing).

### A brief overview of all files
* **TextDataScraper.py**: scrapes and processes meeting minutes from the FOMC website. Generates `text_data`, a dictionary with the meeting minute date as the key and a list of all paragraphs as values.
* **BondRateDataScraper.py**: obtains the 3-month US treasury bond rate from NASDAQ Data Link and calculates the 15-day bond rate difference for all FOMC meeting minute days. Requires `text_data` to obtain dates where FOMC meetings are held. Generates `bond_rate_df`, a dataframe of the 15-day bond rate change on FOMC meeting days. 
* **TextParagraphFeatureBuilder.py**: uses FinBERT and BERTopic to assign topic labels and sentiment scores to paragraphs, and then aggregates them together to generate topic-sentiment features on the document level. Requires `text_data`. Generates `feature_df`, a dataframe containing the topic-sentiment features to be used in the model.
* **Model.py**: trains a MLP, SVM, or LSTM model based on `bond_rate_df` for the target variable and `feature_df` for the features.
* **LSTMmodel.py**: utility class for Model to make pytorch LSTM model work with .fit() and .predict() like sklearn models (i.e. MLP, SVM) so that it approximately works the same way as the two other models in Model.py 
* **predict.py**: combines everything together into one single script to run.

### How to replicate my results
Run predict.py and replace the first two lines with your desired model type and arguments to be fed into the model. 
```py
model_type = "LSTM" # "SVM", "LSTM", or "MLP"
args = {"time_steps": 30, "batch_size": 4} 
```
For MLP and SVM model, arguments that can be used is listen in sklearn's documentation for [MLP](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) and [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). For the LSTM model, the possible parameters are as follows:
- `timesteps`: the number of lookback timesteps the LSTM model uses for its predictions.
- `epochs`: the number of epochs to train the model for.
- `batch_size`: the batch size used for training.
- `hidden_size`: the hidden size for the LSTM model.

A NASDAQ Data Link API is required to scrape the bond rate data. Replace the third line with your own NASDAQ data link API key (get one [here](https://docs.data.nasdaq.com/v1.0/docs/getting-started)):
```py
quandl_authtoken = "YOUR KEY HERE"
```