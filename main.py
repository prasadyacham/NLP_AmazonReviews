# Sentiment Analysis and Topic modelling on Amazon reviews


# Import the required modules

import pandas as pd
import gzip
import numpy as np

import re
import string

# Required for Pre Processing
# HTML Stripping
from bs4 import BeautifulSoup
# Unicode Conversion
import unicodedata


import nltk
from nltk.stem import WordNetLemmatizer

# Model Creation, Vectorization
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import SGDClassifier, LogisticRegression
import joblib
from pathlib import Path

# Visualization
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from wordcloud import WordCloud
from termcolor import colored

import pyLDAvis
from pyLDAvis.sklearn import prepare

# Import the python files
import text_normalizer as tn
import model_evaluation_utils as meu

# Sentiment Analysis
import textblob
from afinn import Afinn
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from datetime import datetime


# Load the data from gzip file 
def load_data(category, file_name):
    df = pd.read_json(
            file_name,
            lines=True,
            compression='gzip'
        )
    
    df['Category'] = category
    
    df = df[['reviewerID','asin','reviewText','Category','unixReviewTime','overall']]
    return df

# Function to comvert the reviews date into DATE format from utc format
def date_conversion(x):
    return datetime.utcfromtimestamp(x).strftime('%Y-%m-%d')


def prepreocess_data(df):
    df['unixReviewTime'] = df['unixReviewTime'].apply(date_conversion)

    # Create a column to define the sentiment based on the user ratings provided
    df['sentiment'] = df['overall'].apply(lambda overall : 'positive' if overall >= 3 else 'negative')
    
    df = df.replace(r'^(\s?)+$', np.nan, regex=True)
    df = df.dropna().reset_index(drop=True)
    
    return df


# Function to remove the puncuations from the document 
def remove_punctuations(document):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filteredtokens =  re.sub(pattern,'', document)
    return filteredtokens

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

def get_stopwords():
    nltk_stopwords  = nltk.corpus.stopwords.words('english')
    nltk_stopwords.remove('no')
    nltk_stopwords.remove('but')
    nltk_stopwords.remove('not')
    new_words = ["absolutley","amazon","anymore","arrive","arrived","box","buy","dollar","like","set","use","used","ha","all","product","does"]
    nltk_stopwords.extend(new_words)
    
    return nltk_stopwords

# Function to remove the stopwords from the token list
def remove_stopwords(tokens):
    stopwords = get_stopwords()
    filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    return filtered_tokens


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

# Function to pre-process the corpus
def normalize_document(data,accented_char_removal=True, html_stripping=True,remove_digits = True,remove_stopword= True,
                       text_lemmatization=True, text_stemmer=False):
    # List to hold the normlized data after tokenizing 
    normalized_data = []
    
    lemmatizer = WordNetLemmatizer()
    
    for doc in data:
        # Remove the digits from the data
        #doc = re.sub(r'[^a-zA-Z\s]', '', doc)
        doc = re.sub(r'\b\d+\b', '', doc)
        
        # Convert each sentence into lower case and strip the extra spaces
        doc = doc.lower()
        doc = doc.strip()
        
        doc = remove_punctuations(doc)
        
        # Normalize unicode characters
        if accented_char_removal:
                doc = remove_accented_chars(doc)

        # remove html tags
        if html_stripping:
                doc = strip_html_tags(doc)

        
        if text_lemmatization:
            word_list = nltk.word_tokenize(doc)
            doc = ' '.join([lemmatizer.lemmatize(w) for w in word_list]) 
        
        # tokenize document into words
        tokens = nltk.word_tokenize(doc)
        
        # remove stopwords out of document
        if remove_stopword:
            no_stopwords_tokens = remove_stopwords(tokens)
        
        # re-create document from filtered tokens
        doc = ' '.join(no_stopwords_tokens)
        
        if text_stemmer:
            ps = nltk.porter.PorterStemmer()
            doc = ' '.join([ps.stem(word) for word in doc.split()])
        
        # append preprocessed document to the list
        normalized_data.append(doc)
        
    return normalized_data


def normalize_data(df):
    norm_corpus = normalize_document(df['reviewText'],accented_char_removal=True, html_stripping=True,remove_digits = True,remove_stopword= True,
                       text_lemmatization=False, text_stemmer=False)
    df['Clean_Reviews'] = norm_corpus
    
    return df

# For each category, we are spitting the data and creating train and test frames.
def data_split(df_category):
    train_data, test_data =train_test_split(df_category, 
                                                         test_size=0.3, random_state=42)
    
    return train_data, test_data

# Function to split the reviews data into training and test data, perform TF_IDF vectorization

def split_vectorization(train_data, test_data):

    #train_data, test_data =data_split(df_category)

    # experiments with different settings results yields the following hyperparameters
    vectorizer = TfidfVectorizer(max_df=.11, 
                                 min_df=.026, 
                                 stop_words='english')

    train_dtm = vectorizer.fit_transform(train_data.Clean_Reviews)
    words = vectorizer.get_feature_names()

    test_dtm = vectorizer.transform(test_data.Clean_Reviews)

    print('TFIDF model: \n\nTrain features shape:', train_dtm.shape, '\nTest features shape:', test_dtm.shape)
    
    return train_dtm , test_dtm , words

def lda(train_data, n_components):
    lda_base = LatentDirichletAllocation(n_components=n_components,
                                     n_jobs=-1,
                                     learning_method='batch',
                                     max_iter=10)
    lda_base.fit(train_data)
    
    joblib.dump(lda_base, model_path / pickle_file_name)
    lda_base = joblib.load(model_path / pickle_file_name) 
    return lda_base

def topic_modelling(lda, words):
    topics_count = lda.components_
    topics_prob = topics_count / topics_count.sum(axis=1).reshape(-1, 1)
    topics = pd.DataFrame(topics_prob.T,
                      index=words,
                      columns=topic_labels)
    #print(len(words))
    #print(topics.shape)
    top_words = {}
    for topic, words_ in topics.items():
        top_words[topic] = words_.nlargest(10).index.tolist()
    df_words = pd.DataFrame(top_words)
    #print (top_words)
    return df_words, topics


def set_topics(topic , typ=1):
    if topic =='Topic 1' and typ==1 :
        return 'Y'
    if topic =='Topic 2' and typ==2 :
        return 'Y'
    if topic =='Topic 3'and typ==3 :
        return 'Y'
    if topic =='Topic 4' and typ==4:
        return 'Y'
    if topic =='Topic 5'and typ==5 :
        return 'Y'
 
    return 'N'

def predict_topic(df,n_components):
    train_dtm, test_dtm, words = split_vectorization(df, df)
    
    lda_base = lda(train_dtm, n_components)
    
    df_topics, topics  = topic_modelling(lda_base,words)
    
    top_words={}
    top_topics =[]
    for topic, words_ in topics.items():
        top_topics += words_.nlargest(15).index.tolist()
        top_words[topic] = words_.nlargest(15).index.tolist()
        
    train_opt_eval = pd.DataFrame(data=lda_base.transform(train_dtm),
                              columns=topic_labels
                             )
    
    df_topics = df.assign(predicted=train_opt_eval.idxmax(axis=1).values)
    
    df_topics['Topic 1'] = df_topics.apply(lambda x: set_topics(x['predicted'],1),axis=1)  
    df_topics['Topic 2'] = df_topics.apply(lambda x: set_topics(x['predicted'],2),axis=1)  
    df_topics['Topic 3'] = df_topics.apply(lambda x: set_topics(x['predicted'],3),axis=1) 
    df_topics['Topic 4'] = df_topics.apply(lambda x: set_topics(x['predicted'],4),axis=1) 
    df_topics['Topic 5'] = df_topics.apply(lambda x: set_topics(x['predicted'],5),axis=1)  

    return df_topics, top_words, top_topics


def get_topic_reviews(topic,df,top_topic):
    for key, val in top_topic.items():
        if topic in val:
            topic_label =key

    topic_df = df[df[topic_label]=='Y']
    return topic_df


def topic_charts(topics):
    fig, ax = plt.subplots(figsize=(10, 14))
    sns.heatmap(topics.sort_values(topic_labels, ascending=False),
                cmap='Blues', ax=ax, cbar_kws={'shrink': .6})
    fig.tight_layout()
    
    
    fig, axes = plt.subplots(nrows=5, sharey=True, sharex=True, figsize=(10, 15))
    for i, (topic, prob) in enumerate(topics.items()):
        sns.distplot(prob, ax=axes[i], bins=100, kde=False, norm_hist=False)
        axes[i].set_yscale('log')
        axes[i].xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    fig.suptitle('Topic Distributions')
    sns.despine()
    fig.tight_layout()

def evaluate(lda_base,train_dtm,test_dtm, train_data):
    train_preds = lda_base.transform(train_dtm)
    train_preds.shape
    train_eval = pd.DataFrame(train_preds, columns=topic_labels, index=train_data['Clean_Reviews'])
    #print(train_eval.head())
    test_preds = lda_base.transform(test_dtm)
    test_eval = pd.DataFrame(test_preds, columns=topic_labels)
    return train_eval, test_eval

def topic_all_data(df):
    vectorizer = CountVectorizer(max_df=.5, 
                             min_df=5,
                             stop_words='english',
                             max_features=2000)
    dtm = vectorizer.fit_transform(df.Clean_Reviews)
    
    lda_all = LatentDirichletAllocation(n_components=n_components,
                                    max_iter=500,
                                    learning_method='batch',
                                    evaluate_every=10,
                                    random_state=42,
                                    verbose=1)
    lda_all.fit(dtm)
    
    joblib.dump(lda_all, model_path /pickle_file_name)
    
    # Fit the LDA model on the vectorized data
    lda_all = joblib.load(model_path / pickle_file_name) 
    
    prepare(lda_all, dtm, vectorizer)
    
    topics_prob = lda_all.components_ / lda_all.components_.sum(axis=1).reshape(-1, 1)
    topics = pd.DataFrame(topics_prob.T,
                      index=vectorizer.get_feature_names(),
                      columns=topic_labels)
    
    print ('Visualize topic-word assocations per document')

    w = WordCloud()
    fig, axes = plt.subplots(nrows=n_components, figsize=(15, 30))
    axes = axes.flatten()
    for t, (topic, freq) in enumerate(topics.items()):
        w.generate_from_frequencies(freq.to_dict())
        axes[t].imshow(w, interpolation='bilinear')
        axes[t].set_title(topic, fontsize=18)
        axes[t].axis('off')
    plt.show()
    
    
def split_data(df):
    cnt = df.shape[0]
    num_rows = int(n * cnt)
    reviews = np.array(df['Clean_Reviews'])
    sentiments = np.array(df['sentiment'])
    train_reviews = reviews[:num_rows]
    train_sentiments = sentiments[:num_rows]
    test_reviews = reviews[num_rows:]
    test_sentiments = sentiments[num_rows:]
    
    return train_reviews, train_sentiments, test_reviews, test_sentiments

def sentiment_analysis(train_reviews, train_sentiments, test_reviews, test_sentiments):
    
    stopwords = ()

    normalized_train_reviews = tn.normalize_corpus(train_reviews, stopwords=stopwords)
    normalized_test_reviews = tn.normalize_corpus(test_reviews, stopwords=stopwords)

    tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0, ngram_range=(1,2),
                     sublinear_tf=True)
    tv_train_features = tv.fit_transform(normalized_train_reviews)
    
    tv_test_features = tv.transform(normalized_test_reviews)
    
    print('TFIDF model:> Train features shape:', tv_train_features.shape, ' Test features shape:', tv_test_features.shape)
    
    return tv_train_features , tv_test_features, train_sentiments, test_sentiments

def sentiment_lr_model(tv_train_features , tv_test_features, train_sentiments, test_sentiments):
    # Logistic Regression model on TF-IDF features
    lr_tfidf_predictions = meu.train_predict_model(classifier=lr, 
                                                   train_features=tv_train_features, train_labels=train_sentiments,
                                                   test_features=tv_test_features, test_labels=test_sentiments)

    #Performance Evaluation for predicted sentiments
    meu.display_model_performance_metrics(true_labels=test_sentiments, predicted_labels=lr_tfidf_predictions,
                                          classes=['positive', 'negative'])
    
def sentiment_svm_model(tv_train_features , tv_test_features, train_sentiments, test_sentiments):
    svm_tfidf_predictions = meu.train_predict_model(classifier=svm, 
                                                train_features=tv_train_features, train_labels=train_sentiments,
                                                test_features=tv_test_features, test_labels=test_sentiments)
    meu.display_model_performance_metrics(true_labels=test_sentiments, predicted_labels=svm_tfidf_predictions,
                                      classes=['positive', 'negative'])  

def predict_sentiment_textblob(test_reviews):
    sentiment_polarity = [textblob.TextBlob(review).sentiment.polarity for review in test_reviews]
    predicted_sentiments = ['positive' if score >= 0.1 else 'negative' for score in sentiment_polarity]
    
    meu.display_model_performance_metrics(true_labels=test_sentiments, predicted_labels=predicted_sentiments, 
                                  classes=['positive', 'negative'])


afn = Afinn(emoticons=True) 

def predict_sentiment_afinn(test_reviews):
    sentiment_polarity = [afn.score(review) for review in test_reviews]
    predicted_sentiments = ['positive' if score >= 1.0 else 'negative' for score in sentiment_polarity]
    meu.display_model_performance_metrics(true_labels=test_sentiments, predicted_labels=predicted_sentiments, 
                                  classes=['positive', 'negative'])

def analyze_sentiment_vader_lexicon(review, 
                                    threshold=0.1,
                                    verbose=False):
    # pre-process text
    review = tn.strip_html_tags(review)
    review = tn.remove_accented_chars(review)
    review = tn.expand_contractions(review)
    
    # analyze the sentiment for review
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(review)
    # get aggregate scores and final sentiment
    agg_score = scores['compound']
    final_sentiment = 'positive' if agg_score >= threshold\
                                   else 'negative'
    if verbose:
        # display detailed sentiment statistics
        positive = str(round(scores['pos'], 2)*100)+'%'
        final = round(agg_score, 2)
        negative = str(round(scores['neg'], 2)*100)+'%'
        neutral = str(round(scores['neu'], 2)*100)+'%'
        sentiment_frame = pd.DataFrame([[final_sentiment, final, positive,
                                        negative, neutral]],
                                        columns=pd.MultiIndex(levels=[['SENTIMENT STATS:'], 
                                                                      ['Predicted Sentiment', 'Polarity Score',
                                                                       'Positive', 'Negative', 'Neutral']], 
                                                              codes=[[0,0,0,0,0],[0,1,2,3,4]]))
        print(sentiment_frame)
    
    return final_sentiment

def predict_sentiment_vader(test_reviews):
    
    predicted_sentiments = [analyze_sentiment_vader_lexicon(review, threshold=0.4, verbose=False) for review in test_reviews]
    meu.display_model_performance_metrics(true_labels=test_sentiments, predicted_labels=predicted_sentiments, 
                                  classes=['positive', 'negative'])


def sentiment_polarity(df):
    df['textblob_polarity'] = df.apply(lambda x : textblob.TextBlob(x['Clean_Reviews']).sentiment.polarity, axis=1)
    df['textblob_sentiment'] = df.apply(lambda x : 'positive' if x['textblob_polarity'] >= 0.1 else 'negative', axis=1 )
    afn = Afinn(emoticons=True) 
    df['afin_polarity'] = df.apply(lambda x : afn.score(x['Clean_Reviews']), axis=1)
    df['afin_sentiment'] = df.apply(lambda x : 'positive' if x['afin_polarity'] >= 1.0 else 'negative', axis=1 )
    df['vader_sentiment'] = df.apply(lambda x : analyze_sentiment_vader_lexicon(x['Clean_Reviews'], threshold=0.4, verbose=False), axis=1 )
    return df



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    
    category = 'Auto'
    if category == 'Auto':
        df_name = 'df_auto_reviews'
        path_name = 'amzn_auto_rw'
        pickle_file_name = 'auto_lda_10_iter.pkl'
        file_name = 'reviews_Automotive_5.json.gz'
    if category == 'Softwares':
        df_name = 'df_sftw_reviews'
        path_name = 'sftw_auto_rw'
        pickle_file_name = 'sftw_lda_10_iter.pkl'
        file_name = 'Software.json.gz'
    if category == 'Giftcard':
        df_name = 'df_gc_reviews'
        path_name = 'gc_auto_rw'
        pickle_file_name = 'gc_lda_10_iter.pkl'
        file_name = 'Gift_Cards.json.gz'
    if category == 'Magazine':
        df_name = 'df_mgzn_reviews'
        path_name = 'mgzn_auto_rw'
        pickle_file_name = 'mgzn_lda_10_iter.pkl'
        file_name = 'Magazine_Subscriptions.json.gz'
    # Load the data into dataframes
    # Import the software Reviews data into a dataframe and select the required columns
    df_softwares = load_data (category, file_name)
    df_reviews = prepreocess_data(df_softwares)
    df_reviews = normalize_data(df_reviews)
    print(df_reviews.head())
    
    train_data, test_data =data_split(df_reviews)
    train_dtm, test_dtm, words = split_vectorization(train_data, test_data)
    
    
    # Set the path to current working directory 
    DATA_DIR = Path().absolute()
    data_path = DATA_DIR / df_name
    print (data_path)
    results_path = Path('results')
    model_path = Path('results', path_name)
    print (model_path)
    if not model_path.exists():
        model_path.mkdir(exist_ok=True, parents=True)
        
    n_components = 5
    topic_labels = [f'Topic {i}' for i in range(1, n_components+1)]
    n = 0.65
    
    lda_base = lda(train_dtm,n_components)
    
    df_topics, topics  = topic_modelling(lda_base,words)
    
    df_topic_review, top_words, top_topics = predict_topic(df_reviews,n_components)
    
    print(df_topic_review.head())
    
    topic = 'wax'
    topic_df = get_topic_reviews(topic,df_topic_review,top_words)
    print(topic_df)
    
    topic_charts(topics)
    
    train_eval, test_eval = evaluate(lda_base, train_dtm,test_dtm, train_data)
    
    topic_all_data(df_reviews)    
    
    nltk.download('stopwords')
    nltk.download('sentiwordnet')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')    
    nltk.download('omw-1.4')
    
    
    train_reviews, train_sentiments, test_reviews, test_sentiments = split_data(df_reviews) 
    
    tv_train_features , tv_test_features, train_sentiments, test_sentiments = sentiment_analysis(train_reviews, train_sentiments, test_reviews, test_sentiments)
    
    lr = LogisticRegression(penalty='l2', max_iter=500, C=1)
    svm = SGDClassifier(loss='hinge', max_iter=100) # linear support vector machine
    
    sentiment_lr_model(tv_train_features , tv_test_features, train_sentiments, test_sentiments)
    sentiment_svm_model(tv_train_features , tv_test_features, train_sentiments, test_sentiments)
    predict_sentiment_textblob(test_reviews)
    predict_sentiment_afinn(test_reviews)
    predict_sentiment_vader(test_reviews)
    
    df = sentiment_polarity(df_reviews)
    df = df[['asin','reviewText','Category','overall','Clean_Reviews','textblob_polarity','textblob_sentiment','afin_polarity','afin_sentiment','vader_sentiment']]
    print(df.head())