#coding=utf-8
# @Time : 21-1-10下午8:10 
# @Author : Honglian WANG

import os
import glob
import io
import pandas
from torchtext import data


class Tweet(data.Dataset):

    urls = None
    name = 'twitter'
    dirname = './data'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, **kwargs):
        """Create an IMDB dataset instance given a path and fields.

        Arguments:
            path: Path to the dataset's highest level directory
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field), ('label', label_field)]
        examples = []

        df = pandas.read_csv(path)
        df = df.dropna(axis=0, how='any')
        df1 = df.values.tolist()


        # if predict=true, file in path has only text but no label
        if 'predict' in kwargs:
            df1 = [[text[0], 'pos'] for text in df1]

        for item in df1:
            examples.append(data.Example.fromlist(item, fields))



        # there is a bug in the super function
        # when calling the super function, ** kwargs is allowed
        # however in the super function, **kwarges is not allowed
        # if you use any **kwargs, you need to clear them first before calling super()
        kwargs = {}
        super(Tweet, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, root='', path = '',
               train='train_clean.csv', test='test_clean.csv', **kwargs):

        # if predict = True, assign test to the path of prediction file.
        if 'predict' in kwargs:
            train = None
            test = kwargs['predict_path']

        return super(Tweet, cls).splits(path=path,
            root=root, text_field=text_field, label_field=label_field,
            train=train, validation=None, test=test, **kwargs)


    @classmethod
    def iters(cls, batch_size=32, device=0, root='.data', vectors=None, **kwargs):
        """Create iterator objects for splits of the IMDB dataset.

        Arguments:
            batch_size: Batch_size
            device: Device to create batches on. Use - 1 for CPU and None for
                the currently active GPU device.
            root: The root directory that contains the imdb dataset subdirectory
            vectors: one of the available pretrained vectors or a list with each
                element one of the available pretrained vectors (see Vocab.load_vectors)

            Remaining keyword arguments: Passed to the splits method.
        """
        TEXT = data.Field()
        LABEL = data.Field(sequential=False)

        train, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)
        LABEL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, test), batch_size=batch_size, device=device)

