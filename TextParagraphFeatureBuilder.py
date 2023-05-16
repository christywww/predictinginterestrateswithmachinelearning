import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from bertopic import BERTopic
from transformers import BertTokenizer, BertForSequenceClassification
from collections import Counter
from nltk import ngrams
from nltk.corpus import stopwords
import math

class ParagraphFeatureBuilder:
    def __init__(self, textdata, topic_count=30):
        self.textdata = textdata
        self.BERTopicModel = None
        self.paragraph_df = None
        self.topic_count = topic_count


    def build_paragraph_df(self):
        # turns textdata into a dataframe (date, text) where each row is a paragraph
        print(datetime.now(), 'Building paragraph df...')
        paragraph_data = []
        for date in self.textdata:
            date_object = datetime.strptime(date[-8:], '%Y%m%d').date()
            for paragraph in self.textdata[date]:
                paragraph_data.append({'date': date_object, 'text': paragraph})
        self.paragraph_df = pd.DataFrame(paragraph_data)

        # remove paragraphs with the exact same text (i.e. statements that are essentially copy & pasted)
        self.paragraph_df = self.paragraph_df.drop_duplicates(subset=['text'], keep=False)

    def keyBERT_GloVe_train(self):
        from gensim.test.utils import datapath, get_tmpfile
        from gensim.models import KeyedVectors
        from gensim.scripts.glove2word2vec import glove2word2vec

        # uses keyBERT to generate a set of keywords
        from keybert import KeyBERT
        from keyphrase_vectorizers import KeyphraseCountVectorizer
        from sklearn.feature_extraction.text import CountVectorizer

        print(datetime.now(), "Extracting keywords with keyBERT...")
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(self.paragraph_df['text'].values, vectorizer=KeyphraseCountVectorizer()) # useMMR=True?

        # load glove model
        glove_file = 'glove.6B.100d.txt'
        word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
        glove2word2vec(glove_file, word2vec_glove_file)
        model = KeyedVectors.load_word2vec_format(word2vec_glove_file)

        # turn keywords into glove embeddings
        keyword_change_dict = {'stockbuilding': 'stock', 'outlookin': 'outlook', 'covid': 'pandemic', 'brexit': 'eu', 'stablecoins': 'bitcoin',
                               'rrps': 'securities', 'tslf': 'securities', 'yct': 'yield', 'mmmf': 'fund', 'mmf': 'fund', 'mmfs': 'fund', 'swaption': 'swaps'}

        keyword_vectors = []
        for keyword_list in keywords: # keyword list: tuples of (keyword, prob) for the top 5 keywords identified in a document
            keyword_vector = []
            for keyword, prob in keyword_list: # keyword: keyword identified (could have multiple words)
                vector_sum = np.zeros(100)
                # each keyword can have multiple words, take an average between the vectors
                for word in keyword.split():
                    valid_word_count = 0
                    if word not in model:
                        if word in keyword_change_dict:
                            vector_sum += model[keyword_change_dict[word]]
                            valid_word_count += 1
                        else:
                            print(word, "|", keyword)
                        continue
                    vector_sum += model[word]
                    valid_word_count += 1
                vector_sum /= valid_word_count
                keyword_vector.append(vector_sum)
            keyword_vectors.append(keyword_vector)
            # this is not working rip
        print('')

    def keyBERT_train(self):
        # use keyBERT to generate a set of keywords
        from keybert import KeyBERT
        from keyphrase_vectorizers import KeyphraseCountVectorizer
        from sklearn.feature_extraction.text import CountVectorizer

        print(datetime.now(), "Extracting keywords with keyBERT...")
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(self.paragraph_df['text'].values, vectorizer=KeyphraseCountVectorizer())

        vocabulary = [k[0] for keyword in keywords for k in keyword]
        vocabulary = list(set(vocabulary))

        vectorizer_model = CountVectorizer(vocabulary=vocabulary)
        topic_model = BERTopic(vectorizer_model=vectorizer_model, verbose=True, nr_topics='auto')
        topics, probs = topic_model.fit_transform(self.paragraph_df['text'].values)
        topic_model.reduce_outliers(self.paragraph_df['text'].values, topics)
        self.paragraph_df['topic'] = topics

        print('')

    def BERTopic_train(self, model_path=None):
        # trains a BERTopic model on self.paragraph_df and saves the model if model_path is given
        print(datetime.now(), 'Building BERTopic model...')
        model = BERTopic(verbose=True, nr_topics='auto')
        topics, probabilities = model.fit_transform(self.paragraph_df['text'].values)
        print(datetime.now(), 'Reducing BERTopic outliers...')
        topics = model.reduce_outliers(self.paragraph_df['text'].values, topics)
        self.paragraph_df['topic'] = topics
        self.BERTopicModel = model

        if model_path is not None:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
                print(datetime.now(), 'Model saved at', model_path)

    def BERTopic_load(self, model_path):
        # load a BERTopic model from a model path
        with open(model_path, 'rb') as f:
            self.BERTopicModel = pickle.load(f)

    def BERTopic_predict(self):
        # generates predictions for BERTopic
        if self.BERTopicModel is None:
            print('No BERTopic model loaded!')
            return
        topics, probs = self.BERTopicModel.transform(self.paragraph_df['text'].values)
        self.paragraph_df['label'] = topics

    def FinBERT_predict(self, filepath=None):
        from tqdm import tqdm
        print(datetime.now(), 'Predicting FinBERT sentiments...')
        model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
        tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        labeldict = {0: 0, 1: 1, 2: -1} # maps labels from FinBert (neutral, positive, negative) to their actual sentiment (0, 1, -1)

        labels = []
        scores = []
        for idx, paragraph in tqdm(self.paragraph_df.iterrows(), total=self.paragraph_df.shape[0]):
            inputs = tokenizer(paragraph['text'], return_tensors="pt")
            outputs = model(**inputs)[0]
            labels.append(labeldict[np.argmax(outputs[0].detach().numpy())])
            # softmax
            result = outputs[0].detach().numpy()
            e_x = np.exp(result- np.max(result))
            scores.append(max(e_x / e_x.sum()))

        self.paragraph_df['sentiment'] = labels
        self.paragraph_df['sent_score'] = scores

        if filepath is not None:
            self.paragraph_df.to_csv(filepath)
            print(datetime.now(), "saved paragraphdf to", filepath)
        print('')

    def build_features(self):
        from collections import defaultdict

        topic_count = self.topic_count

        paragraph_df = self.paragraph_df[self.paragraph_df['topic']<topic_count]

        topic_sentscore_dict = defaultdict(lambda: [{'sents': [], 'word_count':[]} for x in range(topic_count)])
        for idx, para in paragraph_df.iterrows():
            topic_sentscore_dict[para['date']][para['topic']]['sents'].append(para['sentiment'])
            topic_sentscore_dict[para['date']][para['topic']]['word_count'].append(len(para['text'].split()))

        feature_dict = {}
        for minutes in topic_sentscore_dict:
            data = topic_sentscore_dict[minutes]

            total_wordcount = sum([sum(data[topic]['word_count']) for topic in range(topic_count)])
            minutes_features = {}
            for topic in range(topic_count):
                sentiments, word_counts = data[topic]['sents'], data[topic]['word_count']

                sentiment_weights = []
                for sentiment, word_count in zip(sentiments, word_counts):
                    sentiment_weights.append(1-(1/(1+3**(-(word_count-200)/400))))
                minutes_features[str(topic) + '_sentiment'] = sum([sentiment * weight for sentiment, weight in zip(sentiments, sentiment_weights)])/sum(sentiment_weights) if len(sentiments) != 0 else 0

            for topic in range(topic_count):
                minutes_features[str(topic) + '_wordcount'] = sum(data[topic]['word_count']) / total_wordcount

            feature_dict[minutes] = minutes_features

        self.feature_df = pd.DataFrame.from_dict(feature_dict, orient="index")
        self.feature_df = self.feature_df.sort_index()
        print('')

    def process(self, test=False):
        if test: # skip 1-hour finbert step for testing purposes
            self.paragraph_df = pd.read_csv('sent_topic_df.csv')
        else:
            self.build_paragraph_df()
            self.FinBERT_predict(filepath="sent_topic_df.csv")

        if self.BERTopicModel is None:
            self.BERTopic_train()
        else:
            self.BERTopic_predict()

        self.build_features()

        self.feature_df.to_csv('feature_df.csv')

if __name__ == "__main__":
    with open('textdata.pkl', 'rb') as f:
        textdata = pickle.load(f)
    tp = ParagraphFeatureBuilder(textdata)
    # tp.build_paragraph_df()
    # tp.keyBERT_train()
    # tp.paragraph_df.to_csv('paragraph_df.csv')
    tp.process(test=True)
    tp.build_features()
    print('')