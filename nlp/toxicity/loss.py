import logging
logger = logging.getLogger('multilingual')

import numpy as np
import tensorflow as tf

def focal_loss(y_true: tf.Tensor,
               y_pred: tf.Tensor,
               gamma: float = 2.0,
               alpha: float = 0.25,
               reduction: str = 'sum'):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    eps = 1e-7
    y_pred = tf.clip_by_value(y_pred, eps, 1.-eps)

    alpha = tf.ones_like(y_true) * alpha
    alpha = tf.where(tf.equal(y_true, 1.), alpha, 1. - alpha)

    pt = tf.where(tf.equal(y_true, 1.), y_pred, 1. - y_pred)

    loss = -alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt)
    #tf.print('y_pred_min:', tf.reduce_min(y_pred), ', y_pred_max:', tf.reduce_max(y_pred), ', pt:', tf.reduce_min(pt), ', loss:', tf.reduce_max(loss))

    loss = tf.reduce_sum(loss, axis=-1)

    if reduction == 'mean':
        return tf.reduce_mean(loss)
    elif reduction == 'sum':
        return tf.reduce_sum(loss)

    return loss

class ConfusionMatrix(tf.keras.metrics.Metric):
    def __init__(self, class_labels, name=None, **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)

        self.num_classes = len(class_labels)
        self.class_labels = class_labels

        self.cm = self.add_weight('cm', shape=[self.num_classes, self.num_classes], initializer='zeros', dtype=tf.int64,
                aggregation=tf.compat.v1.VariableAggregation.SUM,
                synchronization=tf.VariableSynchronization.ON_READ)

    def update_state(self, y_true, y_pred):
        cm = tf.math.confusion_matrix(labels=y_true, predictions=y_pred, num_classes=self.num_classes, dtype=tf.int64)

        self.cm.assign_add(cm)

    def result(self):
        return self.cm

    def str_result(self):
        cm = self.cm.numpy()

        res = '{}'.format(cm)

        sum_pred = np.sum(cm, axis=0)
        sum_true = np.sum(cm, axis=1)

        for label_idx, name in zip(range(self.num_classes), self.class_labels):
            tp = cm[label_idx][label_idx]

            fp = sum_pred[label_idx] - tp
            fn = sum_true[label_idx] - tp

            rc = tp / (tp + fn + 1e-10) * 100
            pr = tp / (tp + fp + 1e-10) * 100
            f1 = 2 * pr * rc / (pr + rc + 1e-10)

            res += '\n{:15s}: f1: {:4.1f}, precision: {:4.1f}, recall: {:4.1f}, tp: {:3d}, fp: {:3d}, fn: {:3d}'.format(name, f1, pr, rc, tp, fp, fn)

        return res

    def reset_states(self):
        self.cm.assign(tf.zeros_like(self.cm))

class CELoss:
    def __init__(self, is_binary, training):
        self.training = training
        self.is_binary = is_binary

        self.ce_loss = tf.keras.metrics.Mean()
        if is_binary:
            self.acc = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
        else:
            self.acc = tf.keras.metrics.CategoricalAccuracy()

    def reset_states(self):
        self.ce_loss.reset_states()
        self.acc.reset_states()

    def str_result(self):
        return 'ce_loss: {:.3e}, acc: {:.4f}'.format(
                    self.ce_loss.result(),
                    self.acc.result(),
                )

class Metric:
    def __init__(self, toxicity_labels, lang_labels, training, **kwargs):
        self.training = training

        self.total_loss = tf.keras.metrics.Mean()

        self.toxicity = CELoss(is_binary=True, training=training)
        self.lang = CELoss(is_binary=False, training=training)

        self.toxicity_cm = ConfusionMatrix(toxicity_labels)
        self.lang_cm = ConfusionMatrix(lang_labels)

    def reset_states(self):
        self.total_loss.reset_states()

        self.toxicity.reset_states()
        self.toxicity_cm.reset_states()

        self.lang.reset_states()
        self.lang_cm.reset_states()

    def str_result(self):
        if self.training:
            return 'total_loss: {:.3f}, toxicity: {}, lang: {}'.format(
                        self.total_loss.result(),
                        self.toxicity.str_result(),
                        self.lang.str_result(),
                        )
        else:
            return 'toxicity: {}, lang: {}\ntoxicity_cm:\n{}\nlang_cm:\n{}'.format(
                        self.toxicity.str_result(),
                        self.lang.str_result(),

                        self.toxicity_cm.str_result(),
                        self.lang_cm.str_result(),
                        )

class LossMetricAggregator:
    def __init__(self,
                 toxicity_labels,
                 lang_labels,
                 global_batch_size,
                 **kwargs):

        self.global_batch_size = global_batch_size

        label_smoothing = 0.01

        self.toxicity_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing, from_logits=False, reduction=tf.keras.losses.Reduction.NONE, name='toxicity_loss')
        #self.toxicity_loss = lambda y_true, y_pred: focal_loss(y_true, y_pred, reduction='none')
        self.lang_loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing, from_logits=True, reduction=tf.keras.losses.Reduction.NONE, name='lang_loss')

        self.train_metric = Metric(toxicity_labels, lang_labels, training=True, name='train_metric')
        self.eval_metric = Metric(toxicity_labels, lang_labels, training=False, name='eval_metric')

    def str_result(self, training):
        m = self.train_metric
        if not training:
            m = self.eval_metric

        return m.str_result()

    def evaluation_result(self):
        m = self.eval_metric
        acc = m.toxicity.acc.result()

        return acc

    def reset_states(self):
        self.train_metric.reset_states()
        self.eval_metric.reset_states()

    def loss(self, true_toxicity, pred_toxicity, training):
        true_toxicity = tf.cast(true_toxicity, tf.float32)
        pred_toxicity = tf.cast(pred_toxicity, tf.float32)

        pred_toxicity = tf.nn.sigmoid(pred_toxicity)

        ce_toxicity_loss = self.toxicity_loss(true_toxicity, pred_toxicity)

        m = self.train_metric
        if not training:
            m = self.eval_metric

        m.toxicity.ce_loss.update_state(ce_toxicity_loss)
        m.toxicity.acc.update_state(true_toxicity, pred_toxicity)

        if not training:
            pred_toxicity_class = tf.where(pred_toxicity >= 0.5, 1, 0)
            true_toxicity_class = tf.squeeze(true_toxicity, 1)
            m.toxicity_cm.update_state(true_toxicity_class, pred_toxicity_class)

        ce_toxicity_loss = tf.reduce_mean(ce_toxicity_loss)
        return ce_toxicity_loss
