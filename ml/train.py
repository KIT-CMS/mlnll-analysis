import os
import argparse
import pickle

import ROOT
import numpy as np
np.random.seed(1234)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import config as cfg
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(1234)
import tensorflow_probability as tfp


import logging
logger = logging.getLogger('')


@tf.custom_gradient
def count_masking(x, up, down):
    mask = tf.cast(
            tf.cast(x > down, tf.float64) * tf.cast(x <= up, tf.float64),
            tf.float64)
    mask = tf.squeeze(mask)

    def grad(dy):
        width = up - down
        mid = down + 0.5 * width
        sigma = 0.5 * width
        gauss = tf.exp(-1.0 * (x - mid)**2 / 2.0 / sigma**2)
        g = -1.0 * gauss * (x - mid) / sigma**2
        g = tf.squeeze(g) * tf.squeeze(dy)
        return (g, None, None)

    return mask, grad


def setup_logging(output_file, level=logging.DEBUG):
    logger.setLevel(level)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    file_handler = logging.FileHandler(output_file, 'w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def tree2numpy(path, tree, columns):
    df = ROOT.RDataFrame(tree, path)
    return df.AsNumpy(columns)


def build_dataset(path, classes, fold, make_categorical=True, use_class_weights=True):
    columns = cfg.ml_variables + [cfg.ml_weight]
    xs = [] # Inputs
    ys = [] # Targets
    ws = [] # Event weights
    for i, c in enumerate(classes):
        d = tree2numpy(path, c, columns)
        xs.append(np.vstack([np.array(d[k], dtype=np.float64) for k in cfg.ml_variables]).T)
        w = np.array(d[cfg.ml_weight], dtype=np.float64)
        ws.append(w)
        ys.append(np.ones(d[cfg.ml_weight].shape, dtype=np.float64) * i)

    # Stack inputs
    xs = np.vstack(xs)
    logger.debug('Input dataset (shape): {}'.format(xs.shape))

    # Stack targets
    ys = np.hstack(ys)
    logger.debug('Targets, not categorical (shape): {}'.format(ys.shape))

    # Stack weights
    ws = np.hstack(ws)
    logger.debug('Weights, without class weights (shape, sum): {}, {}'.format(ws.shape, np.sum(ws)))

    # Multiply class weights to event weights
    if use_class_weights:
        sum_all = np.sum(ws)
        for i in range(len(classes)):
            mask = ys == i
            ws[mask] = ws[mask] / np.sum(ws[mask]) * sum_all
        logger.debug('Weights, with class weights (shape, sum): {}, {}'.format(ws.shape, np.sum(ws)))

    # Convert targets to categorical
    if make_categorical:
        ys = tf.keras.utils.to_categorical(ys)
        logger.debug('Targets, categorical (shape): {}'.format(ys.shape))

    return xs, ys, ws


def model(x, num_variables, num_classes, fold, reuse=False):
    hidden_nodes = 100
    with tf.variable_scope('model_fold{}'.format(fold), reuse=reuse):
        w1 = tf.get_variable('w1', shape=(num_variables, hidden_nodes), initializer=tf.random_normal_initializer(), dtype=tf.float64)
        b1 = tf.get_variable('b1', shape=(hidden_nodes), initializer=tf.constant_initializer(), dtype=tf.float64)
        w2 = tf.get_variable('w2', shape=(hidden_nodes, num_classes), initializer=tf.random_normal_initializer(), dtype=tf.float64)
        b2 = tf.get_variable('b2', shape=(num_classes), initializer=tf.constant_initializer(), dtype=tf.float64)
    l1 = tf.tanh(tf.add(b1, tf.matmul(x, w1)))
    logits = tf.add(b2, tf.matmul(l1, w2))
    f = tf.sigmoid(logits)
    f = tf.squeeze(f)

    return logits, f, [w1, b1, w2, b2]


def main(args):
    # Build nominal dataset
    classes = cfg.ml_classes + [n + '_ss' for n in cfg.ml_classes if n not in ['ggh', 'qqh']] + ['data_ss']
    x, y, w = build_dataset(os.path.join(args.workdir, 'fold{}.root'.format(args.fold)), classes, args.fold,
                            use_class_weights=False, make_categorical=False)
    x_train, x_val, y_train, y_val, w_train, w_val = train_test_split(x, y, w, test_size=0.25, random_state=1234)
    logger.info('Number of train/val events in nominal dataset: {} / {}'.format(x_train.shape[0], x_val.shape[0]))

    # Scale to expectation in the full dataset
    scale_train = 4.0 / 3.0 * 2.0 # train/test split + two fold
    scale_val = 4.0 * 2.0
    w_train = w_train
    w_val = w_val
    for i, name in enumerate(classes):
        s_train = np.sum(w_train[y_train == i]) * scale_train
        s_val = np.sum(w_val[y_val == i]) * scale_val
        logger.debug('Class / train / val: {} / {} / {}'.format(name, s_train, s_val))

    # Build dataset for systematic shifts
    """
    x_sys, y_sys, w_sys = build_dataset(os.path.join(args.workdir, 'fold{}.root'.format(args.fold)),
            ['htt', 'htt_jecUncRelativeSampleYearUp', 'htt_jecUncRelativeSampleYearDown'], args.fold,
            make_categorical=False, use_class_weights=True)
    x_sys_train, x_sys_val, w_sys_train, w_sys_val = train_test_split(x_sys, w_sys, test_size=0.25, random_state=1234)
    logger.info('Number of train/val events in varied datasets: {} / {}'.format(x_sys_train.shape[0], x_sys_val.shape[0]))
    logger.debug('Sum of weights for nominal/up/down: {} / {} / {}'.format(
        np.sum(w_sys[y_sys == 0]), np.sum(w_sys[y_sys == 1]), np.sum(w_sys[y_sys == 2])))
    """

    # Preprocessing
    preproc = StandardScaler()
    preproc.fit(x_train)
    pickle.dump(preproc, open(os.path.join(args.workdir, 'preproc_fold{}.pickle'.format(args.fold)), 'wb'))
    x_train_preproc = preproc.transform(x_train)
    x_val_preproc = preproc.transform(x_val)
    for i, (var, mean, std) in enumerate(zip(cfg.ml_variables, preproc.mean_, preproc.scale_)):
        logger.info('Variable: %s', var)
        logger.info('Preprocessing parameter (mean, std): %s, %s', mean, std)
        logger.info('Preprocessed data (mean, std): %s, %s', np.mean(x_train_preproc[:, i]), np.std(x_train_preproc[:, i]))

    # Create model
    x_ph = tf.placeholder(tf.float64, shape=(None,len(cfg.ml_variables)))
    logits, f, w_vars = model(x_ph, len(cfg.ml_variables), 1, args.fold)

    # Build NLL loss
    y_ph = tf.placeholder(tf.float64, shape=(None,))
    w_ph = tf.placeholder(tf.float64, shape=(None,))
    s_ph = tf.placeholder(tf.float64, shape=())

    nll = 0.0
    bins = np.array(cfg.analysis_binning)
    mu = tf.constant(1.0, tf.float64)
    nuisances = []
    epsilon = tf.constant(1e-9, tf.float64)
    for i, (up, down) in enumerate(zip(bins[1:], bins[:-1])):
        logger.debug('Add NLL for bin {} with boundaries [{}, {}]'.format(i, down, up))
        up = tf.constant(up, tf.float64)
        down = tf.constant(down, tf.float64)

        # Processes
        mask = count_masking(f, up, down)
        procs = {}
        procs_sumw2 = {}
        for j, name in enumerate(classes):
            proc_w = mask * tf.cast(tf.equal(y_ph, tf.constant(j, tf.float64)), tf.float64) * w_ph
            procs[name] = tf.reduce_sum(proc_w) * s_ph
            procs_sumw2[name] = tf.reduce_sum(tf.square(proc_w)) * s_ph

        # QCD estimation
        procs['qcd'] = procs['data_ss']
        for p in [n for n in cfg.ml_classes if not n in ['ggh', 'qqh']]:
            procs['qcd'] -= procs[p + '_ss']
        procs['qcd'] = tf.maximum(procs['qcd'], 0)
        procs_sumw2['qcd'] = procs['qcd']

        # Nominal signal and background
        sig = 0
        for p in ['ggh', 'qqh']:
            sig += procs[p]

        bkg = 0
        for p in ['ztt', 'zl', 'w', 'tt', 'vv', 'qcd']:
            bkg += procs[p]

        # Bin by bin uncertainties
        sumw2_total = tf.constant(0.0, tf.float64)
        n = tf.constant(0.0, tf.float64)
        nuisances.append(n)
        for p in ['ztt', 'zl', 'w', 'tt', 'vv', 'qcd']:
            sumw2_total += procs_sumw2[p]
        sys = n * tf.sqrt(sumw2_total)

        # Expectations
        obs = sig + bkg
        exp = mu * sig + bkg + sys

        # Likelihood
        nll -= tfp.distributions.Poisson(tf.maximum(exp, epsilon)).log_prob(tf.maximum(obs, epsilon))

    # Nuisance constraints
    for n in nuisances:
        nll -= tfp.distributions.Normal(
                loc=tf.constant(0.0, dtype=tf.float64), scale=tf.constant(1.0, dtype=tf.float64)
                ).log_prob(n)

    # Compute constraint of mu
    def get_constraint(nll, params):
        hessian = [tf.gradients(g, params) for g in tf.unstack(tf.gradients(nll, params))]
        inverse = tf.matrix_inverse(hessian)
        covariance_poi = inverse[0][0]
        constraint = tf.sqrt(covariance_poi)
        return constraint

    loss_fullnll = get_constraint(nll, [mu] + nuisances)
    loss_statsonly = get_constraint(nll, [mu])

    # Add minimization ops
    def get_minimize_op(loss):
        optimizer = tf.train.AdamOptimizer()
        return optimizer.minimize(loss, var_list=w_vars)

    minimize_fullnll = get_minimize_op(loss_fullnll)
    minimize_statsonly = get_minimize_op(loss_statsonly)

    # Train
    config = tf.ConfigProto(intra_op_parallelism_threads=12, inter_op_parallelism_threads=12)
    session = tf.Session(config=config)
    session.run([tf.global_variables_initializer()])
    saver = tf.train.Saver(max_to_keep=1)

    patience = 30
    patience_count = patience
    min_loss = 1e9
    tolerance = 0.001
    step = 0
    validation_steps = 100
    warmup_steps = 500
    while True:
        if step < warmup_steps:
            loss = loss_statsonly
            minimize = minimize_statsonly
            is_warmup = True
        else:
            loss = loss_fullnll
            minimize = minimize_fullnll
            is_warmup = False

        loss_train, _ = session.run([loss, minimize],
                feed_dict={x_ph: x_train_preproc, y_ph: y_train, w_ph: w_train, s_ph: scale_train})

        if step % validation_steps == 0:
            logger.info('Step / patience: {} / {}'.format(step, patience_count))
            logger.info('Train loss: {:.5f}'.format(loss_train))
            loss_val = session.run(loss, feed_dict={x_ph: x_val_preproc, y_ph: y_val, w_ph: w_val, s_ph: scale_val})
            logger.info('Validation loss: {:.5f}'.format(loss_val))

            if is_warmup:
                logger.info('Warmup: {} / {}'.format(step, warmup_steps))
            else:
                if min_loss > loss_val and np.abs(min_loss - loss_val) / min_loss > tolerance:
                    min_loss = loss_val
                    patience_count = patience
                    path = saver.save(session, os.path.join(args.workdir, 'model_fold{}/model.ckpt'.format(args.fold)), global_step=step)
                    logger.info('Save model to {}'.format(path))
                else:
                    patience_count -= 1

                if patience_count == 0:
                    logger.info('Stop training')
                    break

        step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    parser.add_argument('fold', type=int, help='Training fold')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'ml_train_fold{}.log'.format(args.fold)), logging.DEBUG)
    main(args)
