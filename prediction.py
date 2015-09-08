__author__ = 'sandip'


import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDRegressor
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cross_validation import KFold



class FactorExtractor:
  def __init__(self, factor):
    self.factor = factor

  def transform(self, data):
    return [{self.factor: self.normalize(tt)} for tt in data[self.factor]]

  def fit(self, *_):
    return self

  def normalize(self, tag):
    if type(tag) != str: tag = '_MISSING_'
    return tag


class TextExtractor:
  def __init__(self, column):
    self.column = column

  def transform(self, data):
    return np.asarray(data[self.column])

  def fit(self, *_):
    return self


class IndustryExtractor:
  def transform(self, data):
    return np.asarray(data[['industry']]).astype('str')

  def fit(self, *_):
    return self


class CompanyExtractor:
  def transform(self, data):
    return np.asarray(data[['company']]).astype('str')

  def fit(self, *_):
    return self


class LocationExtractor:
  def transform(self, data):
    return np.asarray(data[['location']]).astype('str')

  def fit(self, *_):
    return self


class FunctionExtractor:
  def transform(self, data):
    return np.asarray(data[['function']]).astype('str')

  def fit(self, *_):
    return self


title_ngrams_featurizer = Pipeline([
  ('desc_extractor',    TextExtractor('title')),
  ('count_vectorizer',  CountVectorizer(ngram_range = (1, 3), stop_words = 'english')),
  ('tfidf_transformer', TfidfTransformer())
])

desc_ngrams_featurizer = Pipeline([
  ('desc_extractor',    TextExtractor('desc')),
  ('count_vectorizer',  CountVectorizer(ngram_range = (2, 4), stop_words = 'english')),
  ('tfidf_transformer', TfidfTransformer())
])

industry_featurinzer = Pipeline([
    ('industry_extractor',     IndustryExtractor()),
    ('one_hot_converter',       OneHotEncoder())
])

company_featurinzer = Pipeline([
    ('company_extractor',     CompanyExtractor()),
    ('one_hot_converter',       OneHotEncoder())
])

location_featurinzer = Pipeline([
    ('location_extractor',     FactorExtractor('location')),
    ('dict_vectorizer',     DictVectorizer(sparse = False))
])

function_featurinzer = Pipeline([
    ('function_extractor',     FunctionExtractor()),
    ('one_hot_converter',       OneHotEncoder())
])

features = FeatureUnion([
  ('title_features',            title_ngrams_featurizer),
  ('desc_features',             desc_ngrams_featurizer),
  # ('industry_features',         industry_featurinzer),
  ('company_features',          company_featurinzer),
  ('location_features',         location_featurinzer)
  # ('function_features',         function_featurinzer)
])

#Tried different predictors including ExtraClassifier, Randomforestclassifier, Linear SVM with different params

predictor = ExtraTreesClassifier(n_estimators=50)

pipeline = Pipeline([
  ('feature_union',  features),
  ('predictor',      predictor)
])


#Training phase

whole_data = pd.load('complete_job')
train_data = whole_data[0:10000]
test_data = whole_data[9001:10000]
feature = ['title', 'desc', 'company', 'location']
predictable = 'function'
train_target = train_data[predictable]
train_input = train_data[feature]
test_input = test_data[feature]
test_target = test_data[predictable]

print "training start"
pipeline.fit(train_input,train_target)
predictions = pipeline.predict(test_input)

score = pipeline.score(test_input, test_target)
print zip(predictions, test_target)


#test prediction for incomplete jobs


#final test
final_test_data = pd.load('incomplete_job')
final_test_input = final_test_data[feature]
final_prediction = pipeline.predict(final_test_input)
print zip(final_test_data['id'],final_prediction)

result_columns = ['id','function']

result_file = pd.DataFrame(columns=result_columns)

result_file['id'] = final_test_data['id']
result_file['function'] = final_prediction

print result_file.info()

result_file.to_csv('final_result_function.csv',sep=",")
print score