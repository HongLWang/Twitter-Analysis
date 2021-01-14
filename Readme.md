#### Jupyter notebooks

```
There are four jupyter notebook in this directory.
Twitter_Dataset_Analysis.ipynb
Tweet_Topic_Modeling.ipynb
Twitter_Community_Detection_Visualization.ipynb
Tweet_Sentiment_Analysis.ipynb

Details can be found in the notebook.
```

#### Datasets

```
All data needed to run these notebooks can be downloaded from https://drive.google.com/drive/folders/1eDYECK9UnDqhuy7KkA6ISK4w8mD18GdQ?usp=sharing

```


#### Preprocessing.py

```
This file defines the text preprocessing function. The preprocessing steps are consisted of
1) remove url 2) remove html 3) change words to lower case
4) remove "@" tag  5) remove digits 6)remove punctuations  
7) convert emojis and emoticons 8) remove stopwords 9) remove whitespace
10) lemmatize words
```

#### Tweets.py

```
This file is a similar file to torchtext.dataset.imdb,with changes on
1) ways to load datset
2) add kwargs to allow producing prediction dataset
```

#### attractor_cache/

```
This directory saves the caches then calling Attractor to perform community detection
```

#### sentAnalysis_data/

```
This directory provide training and testing data for Twitter sentiment classification. An unlabeled "prediction.csv" is also provided for prediction task
```

#### word_clouds/

```
This directory is the path to save word cloud html files.
```

#### Topic_tweets/

```
This is the directory to save twitters that belong to the same topic.
```

