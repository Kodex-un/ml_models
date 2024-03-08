import argparse
import cv2
import logging
import os
import random
import re
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import tensorflow_addons as tfa

use_horovod = False

if use_horovod:
    import horovod.tensorflow as hvd
else:
    class Hvd:
        def __init__(self):
            pass

        def init(self):
            pass

        def size(self):
            return 1
        def rank(self):
            return 0
        def local_rank(self):
            return 0

        def DistributedGradientTape(self, tape):
            return tape

        def broadcast_variables(self, variables=None, root_rank=0):
            pass

hvd = Hvd()

logger = logging.getLogger('multilingual')

import encoder
import load_dataset
import loss

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, required=True, help='Number of images to process in a batch')
parser.add_argument('--batch_size', type=int, default=24, help='Number of images to process in a batch')
parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to run')
parser.add_argument('--num_epochs_stage0', type=int, default=400, help='Number of epochs to run in the stage0 phase')
parser.add_argument('--train_dir', type=str, required=True, help='Path to train directory, where graph will be stored')
parser.add_argument('--base_checkpoint', type=str, help='Load base model weights from this file')
parser.add_argument('--use_good_checkpoint', action='store_true', help='Recover from the last good checkpoint when present')
parser.add_argument('--model_name', type=str, default='xlm-roberta-large', choices=['xlm-roberta-large'], help='Model name')
parser.add_argument('--languages', type=str, default='tr,it,es,ru,fr,pt', help='Languages')
parser.add_argument('--initial_learning_rate_transformer', default=5e-6, type=float,
        help='Initial transformer\'s learning rate (will be multiplied by the number of nodes in the distributed strategy)')
parser.add_argument('--initial_learning_rate_head', default=1e-3, type=float,
        help='Initial head\'s learning rate (will be multiplied by the number of nodes in the distributed strategy)')
parser.add_argument('--min_learning_rate_transformer', default=5e-7, type=float, help='Minimal transformer\'s learning rate')
parser.add_argument('--min_learning_rate_head', default=1e-6, type=float, help='Minimal head\'s learning rate')
parser.add_argument('--print_per_train_steps', default=20, type=int, help='Print train stats per this number of steps(batches)')
parser.add_argument('--min_eval_metric', default=0.2, type=float, help='Minimal evaluation metric to start saving models')
parser.add_argument('--epochs_lr_update', default=20, type=int, help='Maximum number of epochs without improvement used to reset or decrease learning rate')
parser.add_argument('--use_fp16', action='store_true', help='Whether to use fp16 training/inference')
parser.add_argument('--steps_per_train_epoch', default=400, type=int, help='Number of steps per train run')
parser.add_argument('--steps_per_eval_epoch', default=40, type=int, help='Number of steps per evaluation run')
parser.add_argument('--reset_on_lr_update', action='store_true', help='Whether to reset to the best model after learning rate update')
parser.add_argument('--optimizer', type=str, choices=['adam', 'radam+lookahead'], default='adam', help='Whether to reset to the best model after learning rate update')

def train():
    checkpoint_dir = os.path.join(FLAGS.train_dir, 'checkpoints')
    good_checkpoint_dir = os.path.join(checkpoint_dir, 'good')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(good_checkpoint_dir, exist_ok=True)

    handler = logging.FileHandler(os.path.join(checkpoint_dir, 'train.log.{}'.format(hvd.rank())), 'a')
    handler.setFormatter(__fmt)
    logger.addHandler(handler)

    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    num_replicas = hvd.size()

    dtype = tf.float32
    if FLAGS.use_fp16:
        dtype = tf.float16
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

    FLAGS.initial_learning_rate_transformer *= num_replicas
    FLAGS.initial_learning_rate_head *= num_replicas

    if hvd.rank() == 0:
        logdir = os.path.join(FLAGS.train_dir, 'logs')
        writer = tf.summary.create_file_writer(logdir)
        writer.set_as_default()

    global_step = tf.Variable(0, dtype=tf.int64, name='global_step', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
    epoch_var = tf.Variable(0, dtype=tf.float32, name='epoch_number', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

    learning_rate_transformer = tf.Variable(FLAGS.initial_learning_rate_transformer, dtype=tf.float32, name='learning_rate/transformer')
    learning_rate_head = tf.Variable(FLAGS.initial_learning_rate_head, dtype=tf.float32, name='learning_rate/head')

    def create_optimizer(name, lr, min_lr):
        if FLAGS.optimizer == 'adam':
            opt = tf.optimizers.Adam(lr=lr)
        elif FLAGS.optimizer == 'radam+lookahead':
            opt = tfa.optimizers.RectifiedAdam(lr=lr, min_lr=min_lr)
            opt = tfa.optimizers.Lookahead(opt, sync_period=6, slow_step_size=0.5)

        if FLAGS.use_fp16:
            opt = mixed_precision.LossScaleOptimizer(opt, loss_scale='dynamic')

        return opt

    opt_tr = create_optimizer('transformer', learning_rate_transformer, FLAGS.min_learning_rate_transformer)
    opt_head = create_optimizer('head', learning_rate_head, FLAGS.min_learning_rate_head)

    model, tokenizer = encoder.create_encoder(FLAGS.model_name)

    langs = FLAGS.languages.split(',')
    dataset_wrapper = load_dataset.Dataset(FLAGS.dataset_dir, tokenizer, split_to=hvd.size(), rank=hvd.rank(), batch_size=FLAGS.batch_size, langs=langs)

    #dummy_input = tf.ones((FLAGS.batch_size, image_size, image_size, 3), dtype=dtype)
    #_ = model(dummy_input, training=True)

    #line_length = 128
    #model.summary(line_length=line_length, print_fn=lambda line: logger.info(line))

    def calc_epoch_steps(num_examples):
        return (num_examples + FLAGS.batch_size - 1) // (FLAGS.batch_size)

    steps_per_train_epoch = FLAGS.steps_per_train_epoch
    if steps_per_train_epoch < 0:
        steps_per_train_epoch = calc_epoch_steps(len(dataset_wrapper.train_df))

    steps_per_eval_epoch = FLAGS.steps_per_eval_epoch
    if steps_per_eval_epoch < 0:
        steps_per_eval_epoch = calc_epoch_steps(len(dataset_wrapper.eval_df))

    logger.info('model_name: {}, steps_per_train_epoch: {}, steps_per_eval_epoch: {}'.format(
        FLAGS.model_name,
        steps_per_train_epoch, steps_per_eval_epoch))

    checkpoint = tf.train.Checkpoint(step=global_step, epoch=epoch_var, optimizer_head=opt_head, optimizer_transformer=opt_tr, model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=20)

    restore_path = None

    if FLAGS.use_good_checkpoint:
        restore_path = tf.train.latest_checkpoint(good_checkpoint_dir)
        if restore_path:
            status = checkpoint.restore(restore_path)
            logger.info("Restored from good checkpoint {}, global step: {}".format(restore_path, global_step.numpy()))

    if not restore_path:
        status = checkpoint.restore(manager.latest_checkpoint)

        if manager.latest_checkpoint:
            logger.info("Restored from {}, global step: {}, epoch: {}".format(manager.latest_checkpoint, global_step.numpy(), epoch_var.numpy()))
            restore_path = manager.latest_checkpoint
        else:
            logger.info("Initializing from scratch, no latest checkpoint")

            if FLAGS.base_checkpoint:
                base_checkpoint = tf.train.Checkpoint(step=global_step, epoch=epoch_var, optimizer=opt, model=model.body)
                status = base_checkpoint.restore(FLAGS.base_checkpoint)
                status.expect_partial()

                saved_path = manager.save()
                logger.info("Restored base model from external checkpoint {} and saved object-based checkpoint {}".format(FLAGS.base_checkpoint, saved_path))
                exit(0)

    toxicity_labels = ["non-toxic", "toxic"]
    lang_labels = langs
    metric = loss.LossMetricAggregator(toxicity_labels, lang_labels, FLAGS.batch_size * num_replicas)

    @tf.function
    def eval_step(text_tokens, true_toxicity_labels):
        pred_toxicity = model(text_tokens, training=False)
        toxicity_loss = metric.loss(true_toxicity_labels, pred_toxicity, training=False)
        total_loss = toxicity_loss
        metric.eval_metric.total_loss.update_state(total_loss)
        return total_loss

    @tf.function
    def train_step(text_tokens, true_toxicity_labels):
        with tf.GradientTape(persistent=True) as tape:
            pred_toxicity = model(text_tokens, training=True)
            toxicity_loss = metric.loss(true_toxicity_labels, pred_toxicity, training=True)
            total_loss = toxicity_loss
            metric.train_metric.total_loss.update_state(total_loss)

            if FLAGS.use_fp16:
                scaled_total_loss = opt.get_scaled_loss(total_loss)

        tape = hvd.DistributedGradientTape(tape)

        custom_variables = []
        transformer_variables = [v for v in model.trainable_variables
                                    if (('pooler' not in v.name) and
                                        ('custom' not in v.name))]
        for var in model.trainable_variables:
            if 'custom_head' in var.name:
                custom_variables.append(var)

        if FLAGS.use_fp16:
            custom_gradients = tape.gradient(scaled_total_loss, custom_variables)
            transformer_gradients = tape.gradient(scaled_total_loss, transformer_variables)

            custom_gradients = opt.get_unscaled_gradients(custom_gradients)
            transformer_gradients = opt.get_unscaled_gradients(transformer_gradients)
        else:
            custom_gradients = tape.gradient(total_loss, custom_variables)
            transformer_gradients = tape.gradient(total_loss, transformer_variables)

        del tape

        opt_tr.apply_gradients(zip(transformer_gradients, transformer_variables))
        opt_head.apply_gradients(zip(custom_gradients, custom_variables))

        global_step.assign_add(1)
        return total_loss

    def run_epoch(name, dataset, step_func, max_steps, broadcast_variables=False):
        if name == 'train':
            m = metric.train_metric
        else:
            m = metric.eval_metric

        step = 0
        def log_progress():
            if name == 'train':
                logger.info('{}: step: {} {}/{}: {}'.format(
                    int(epoch_var.numpy()), global_step.numpy(), step, max_steps,
                    metric.str_result(True),
                    ))

            if hvd.rank() == 0:
                if name == 'train':
                    tf.summary.scalar(f'{name}/lr_transformer', learning_rate_transformer, step=global_step)
                    tf.summary.scalar(f'{name}/lr_head', learning_rate_head, step=global_step)

                    tf.summary.scalar(f'{name}/epoch', epoch, step=global_step)

                tf.summary.scalar('{name}/total_loss', m.total_loss.result(), step=global_step)

                tf.summary.scalar(f'{name}/ce_loss/toxicity', m.toxicity.ce_loss.result(), step=global_step)
                tf.summary.scalar(f'{name}/ce_loss/lang', m.lang.ce_loss.result(), step=global_step)

                tf.summary.scalar(f'{name}/accuracy/toxicity', m.toxicity.acc.result(), step=global_step)
                tf.summary.scalar(f'{name}/accuracy/lang', m.lang.acc.result(), step=global_step)

        first_batch = True
        for text_tokens, true_toxicity_labels in dataset:
            total_loss = step_func(text_tokens, true_toxicity_labels)

            if name == 'train':
                if first_batch and broadcast_variables:
                    logger.info('broadcasting initial variables')
                    hvd.broadcast_variables(model.variables, root_rank=0)
                    hvd.broadcast_variables(opt_tr.variables(), root_rank=0)
                    hvd.broadcast_variables(opt_head.variables(), root_rank=0)
                    first_batch = False

                if (step % FLAGS.print_per_train_steps == 0) or np.isnan(total_loss.numpy()):
                    log_progress()

                    if np.isnan(total_loss.numpy()):
                        exit(-1)


            step += 1
            if step >= max_steps:
                break

        log_progress()

        return step

    best_metric = 0
    best_saved_path = None
    num_epochs_without_improvement = 0
    initial_learning_rate_multiplier = 0.2
    learning_rate_multiplier = initial_learning_rate_multiplier

    def validation_metric():
        return metric.evaluation_result()

    if hvd.rank() == 0:
        if restore_path:
            metric.reset_states()
            logger.info('there is a checkpoint {}, running initial validation'.format(restore_path))

            eval_steps = run_epoch('eval', dataset_wrapper.eval_ds, eval_step, steps_per_eval_epoch, broadcast_variables=False)
            best_metric = validation_metric()
            best_saved_path = restore_path
            logger.info('initial validation: {}, metric: {:.3f}, path: {}'.format(metric.str_result(False), best_metric, best_saved_path))

        if best_metric < FLAGS.min_eval_metric:
            logger.info('setting minimal evaluation metric {:.3f} -> {} from command line arguments'.format(best_metric, FLAGS.min_eval_metric))
            best_metric = FLAGS.min_eval_metric

    num_vars = len(model.trainable_variables)
    num_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])

    logger.info('nodes: {}, checkpoint_dir: {}, model: {}, model trainable variables/params: {}/{}'.format(
        num_replicas, checkpoint_dir, FLAGS.model_name,
        num_vars, int(num_params)))

    train_datasets = [dataset_wrapper.train_ds, dataset_wrapper.eval_train_ds]
    eval_datasets = [dataset_wrapper.eval_ds, dataset_wrapper.eval_eval_df]

    steps_per_train_epoch = [steps_per_train_epoch, calc_epoch_steps(len(dataset_wrapper.eval_train_df))]
    steps_per_eval_epoch = [steps_per_eval_epoch, calc_epoch_steps(len(dataset_wrapper.eval_eval_df))]
    stage = 0

    learning_rate_transformer.assign(FLAGS.initial_learning_rate_transformer)
    learning_rate_head.assign(FLAGS.initial_learning_rate_head)
    for epoch in range(FLAGS.num_epochs):
        metric.reset_states()
        want_reset = False

        stage = stage % len(train_datasets)

        train_steps = run_epoch('train', train_datasets[stage], train_step, steps_per_train_epoch[stage], (epoch == 0))
        eval_steps = run_epoch('eval', eval_datasets[stage], eval_step, steps_per_eval_epoch[stage], broadcast_variables=False)

        epoch_var.assign_add(1)

        new_lr_tr = learning_rate_transformer.numpy()
        new_lr_head = learning_rate_head.numpy()

        new_metric = validation_metric()

        logger.info('epoch: {}/{}, stage: {}, train: steps: {}, lr_tr: {:.2e}, lr_head: {:.2e}, train: {}, eval: {}, val_metric: {:.4f}/{:.4f}'.format(
            int(epoch_var.numpy()), num_epochs_without_improvement, stage, global_step.numpy(),
            learning_rate_transformer.numpy(), learning_rate_head.numpy(),
            metric.str_result(True), metric.str_result(False),
            new_metric, best_metric))

        if hvd.rank() == 0:
            saved_path = manager.save()

        if new_metric > best_metric:
            if hvd.rank() == 0:
                best_saved_path = checkpoint.save(file_prefix='{}/ckpt-{:.4f}'.format(good_checkpoint_dir, new_metric))

            logger.info("epoch: {}/{}, stage: {}, global_step: {}, saved checkpoint: {}, eval metric: {:.4f} -> {:.4f}: {}".format(
                int(epoch_var.numpy()), num_epochs_without_improvement, stage, global_step.numpy(), best_saved_path, best_metric, new_metric, metric.str_result(False)))

            best_metric = new_metric

            num_epochs_without_improvement = 0
            learning_rate_multiplier = initial_learning_rate_multiplier
        else:
            num_epochs_without_improvement += 1


        if num_epochs_without_improvement >= FLAGS.epochs_lr_update:
            if learning_rate_head > FLAGS.min_learning_rate_head:
                def lr_update(learning_rate, min_learning_rate):
                    new_lr = learning_rate.numpy() * learning_rate_multiplier
                    if new_lr < min_learning_rate:
                        new_lr = min_learning_rate
                    return new_lr

                new_lr_tr = lr_update(learning_rate_transformer, FLAGS.min_learning_rate_transformer)
                new_lr_head = lr_update(learning_rate_head, FLAGS.min_learning_rate_head)

                if FLAGS.reset_on_lr_update:
                    want_reset = True

                logger.info('epoch: {}/{}, stage: {}, global_step: {}, epochs without metric improvement: {}, best metric: {:.5f}, updating learning rate: transformer: {:.2e} -> {:.2e}, head: {:.2e} -> {:.2e}, will reset: {}'.format(
                    int(epoch_var.numpy()), num_epochs_without_improvement, stage, global_step.numpy(), num_epochs_without_improvement, best_metric,
                    learning_rate_transformer.numpy(), new_lr_tr,
                    learning_rate_head.numpy(), new_lr_head,
                    want_reset))

                num_epochs_without_improvement = 0
                if learning_rate_multiplier > 0.1:
                    learning_rate_multiplier /= 2


            elif num_epochs_without_improvement >= FLAGS.epochs_lr_update:
                new_lr_tr = FLAGS.initial_learning_rate_transformer
                new_lr_head = FLAGS.initial_learning_rate_head
                want_reset = True

                logger.info('epoch: {}/{}, stage: {}, global_step: {}, epochs without metric improvement: {}, best metric: {:.5f}, resetting learning rate: transformer: {:.2e} -> {:.2e}, head: {:.2e} -> {:.2e}, will reset: {}'.format(
                    int(epoch_var.numpy()), num_epochs_without_improvement, stage, global_step.numpy(), num_epochs_without_improvement, best_metric,
                    learning_rate_transformer.numpy(), new_lr_tr,
                    learning_rate_head.numpy(), new_lr_head,
                    want_reset))

                num_epochs_without_improvement = 0
                learning_rate_multiplier = initial_learning_rate_multiplier

                if epoch > FLAGS.num_epochs_stage0:
                    stage = 1
                    logger.info('epoch: {}, max number of stage0 epochs: {}, switching to stage1'.format(
                        int(epoch_var.numpy()), FLAGS.num_epochs_stage0,
                        ))


        if want_reset:
            restore_path = tf.train.latest_checkpoint(good_checkpoint_dir)
            if restore_path:
                epoch_num = epoch_var.numpy()
                step_num = global_step.numpy()
                logger.info('epoch: {}/{}, stage: {}, global_step: {}, best metric: {:.5f}, learning rate: transformer: {:.2e} -> {:.2e}, head: {:.2e} -> {:.2e}, restoring best checkpoint: {}'.format(
                    int(epoch_var.numpy()), num_epochs_without_improvement, stage, global_step.numpy(), best_metric,
                    learning_rate_transformer.numpy(), new_lr_tr,
                    learning_rate_head.numpy(), new_lr_head,
                    best_saved_path))

                checkpoint.restore(best_saved_path)

                epoch_var.assign(epoch_num)
                global_step.assign(step_num)

        # update learning rate even without resetting model
        learning_rate_transformer.assign(new_lr_tr)
        learning_rate_head.assign(new_lr_head)


if __name__ == '__main__':
    hvd.init()

    # deterministic seed
    random_seed = 0

    random.seed(random_seed)
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    np.set_printoptions(formatter={'float': '{:0.4f}'.format, 'int': '{:4d}'.format}, linewidth=250, suppress=True, threshold=np.inf)

    logger.propagate = False
    logger.setLevel(logging.INFO)
    __fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
    if hvd.rank() == 0:
        __handler = logging.StreamHandler()
        __handler.setFormatter(__fmt)
        logger.addHandler(__handler)

    FLAGS = parser.parse_args()

    try:
        train()
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()

        logger.error("got error: {}".format(e))

        import traceback

        lines = traceback.format_exc().splitlines()
        for l in lines:
            logger.info(l)

        traceback.print_exception(exc_type, exc_value, exc_traceback)
        exit(-1)
