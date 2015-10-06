# job-categorization---Experteer-GmBH
* **Data**- The data is a xml file which contains job description, title, location, company, career level, job function.
The idea is to predict career level and function the job requirement belongs to, using machine learning techniques
* **Data preparation** - load_data.py contains the data extraction from the xml to pandas dataframe to be used by learning
algorithms. Since the text regarding career level and function is present around some keywords, only certain part of 
job description is extracted(e.g text around the keyword "Requirements" and "Experience").

* **prediction**- In the next step, prediction.py contains how ML is used on the pandas dataframe created.
  * For the text features, tf-idf created features created using scikit countvectorizer and Tfidfvectorizer.
  * for the categorical features, I used one-hop-encoding to create numeric features.
  * All the features are merged using scikit Featureunion methods, and then tried different ML methods to get the best accuracy.
  * For the hyperparameters, Scikit grid search is used.
