import logging
logger = logging.getLogger('multilingual')

import tensorflow as tf

import transformers as tr

class Head(tf.keras.layers.Layer):
    def __init__(self, dropout_rate, name='custom_head'):
        super().__init__(name=name)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.out = tf.keras.layers.Dense(1, name='toxicity_head')

    def __call__(self, inputs, training):
        x = self.dropout(inputs, training)
        x = self.out(x)
        return x

class Roberta(tf.keras.Model):
    def __init__(self, pretrained_model, dropout_rate=0.3):
        super().__init__()

        self.features = tr.TFRobertaModel.from_pretrained(pretrained_model)
        self.head = Head(dropout_rate=dropout_rate, name='custom_head')

    def __call__(self, inputs, training):
        x = self.features(inputs, training=training)
        x = x[0][:, 0, :]
        x = self.head(x, training)
        return x

def create_encoder(model_name):
    if model_name == 'xlm-roberta-large':
        pretrained_model = 'jplu/tf-xlm-roberta-large'

        tokenizer = tr.AutoTokenizer.from_pretrained(pretrained_model)
        model = Roberta(pretrained_model=pretrained_model)

        return model, tokenizer
    else:
        raise ValueError(f'unsupported model name {model_name}')
