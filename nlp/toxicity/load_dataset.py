import os
import logging

logger = logging.getLogger('multilingual')

import numpy as np
import pandas as pd
import tensorflow as tf

class Dataset:
    def __init__(self, dataset_dir, tokenizer, split_to=1, rank=0, batch_size=128, langs=['tr', 'it', 'es', 'ru', 'fr', 'pt'], columns=['comment_text', 'toxic'], maxlen=192):
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer

        self.columns = columns
        self.langs = langs

        self.train_df = self.load_jigsaw_trans(langs, columns)
        self.eval_df = self.load_jigsaw_validation(langs)

        self.train_df = np.array_split(self.train_df, split_to)[rank]

        self.train_ds, self.eval_ds = self.create_dataset_from_dfs(self.train_df, self.eval_df, batch_size=batch_size, maxlen=maxlen)
        
        self.split_validation_data_for_train(ratio=0.9)

        self.eval_train_ds, self.eval_eval_ds = self.create_dataset_from_dfs(self.eval_train_df, self.eval_eval_df, batch_size=batch_size, maxlen=maxlen)

    def create_dataset_from_dfs(self, train_df, eval_df, batch_size, maxlen):
        train_values = self.encode(train_df.comment_text.values, maxlen=maxlen)
        eval_values = self.encode(eval_df.comment_text.values, maxlen=maxlen)

        train_labels = train_df.toxic.values.reshape(-1, 1)
        eval_labels = eval_df.toxic.values.reshape(-1, 1)

        train_ds = self.create_dataset(train_values, train_labels, training=True, batch_size=batch_size)
        eval_ds = self.create_dataset(eval_values, eval_labels, training=False, batch_size=batch_size)

        return train_ds, eval_ds

    def load_jigsaw_validation(self, langs):
        df = pd.read_csv(os.path.join(self.dataset_dir, 'validation.csv'))
        df = df.loc[np.where(df.lang.isin(langs))][self.columns]
        return df

    def split_validation_data_for_train(self, ratio=0.9):
        indexes = np.random.rand(len(self.eval_df))
        train_indexes = indexes > ratio
        eval_indexes = ~train_indexes

        self.eval_train_df = self.eval_df.loc[train_indexes, :]
        self.eval_eval_df = self.eval_df.loc[eval_indexes, :]

    def load_jigsaw_trans(self, langs, columns):
        dfs = []
        for lang in langs:
            fn = f'jigsaw-toxic-comment-train-google-{lang}-cleaned.csv'
            fn = os.path.join(self.dataset_dir, fn)

            df = pd.read_csv(fn, usecols=columns)
            sampled_df = self.downsample(df)
            dfs.append(sampled_df)

            logger.info(f'dataset: "{lang}", entries total: {len(df)}, sampled: {len(sampled_df)}')

        df = pd.concat(dfs)
        logger.info(f'loaded datasets: {len(langs)}, entries: {len(df)}')
        return df

    def downsample(self, df):
        ds_df= pd.concat([
            df.query('toxic==1'),
            df.query('toxic==0').sample(sum(df.toxic))
        ])
        
        return ds_df

    def encode(self, texts, maxlen):
        enc = self.tokenizer.batch_encode_plus(texts, 
                    #return_attention_masks=False, 
                    return_token_type_ids=False,
                    pad_to_max_length=True,
                    truncation=True,
                    max_length=maxlen)
        
        return np.array(enc['input_ids'])

    def create_dataset(self, x, y=None, training=False, batch_size=512):
        dataset = tf.data.Dataset.from_tensor_slices(x)

        if y is not None:
            dataset_y = tf.data.Dataset.from_tensor_slices(y)
            dataset = tf.data.Dataset.zip((dataset, dataset_y))
            
        if training:
            dataset = dataset.shuffle(len(x)).repeat()

        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        #dist_dataset = strategy.experimental_distribute_dataset(dataset)

        return dataset
