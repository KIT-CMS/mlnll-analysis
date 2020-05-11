import os
import argparse

import ROOT
from utils import config as cfg

import logging
logger = logging.getLogger('')


# Global lsit of chains to keep everything alive
chains = []


def setup_logging(output_file, level=logging.DEBUG):
    logger.setLevel(level)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    file_handler = logging.FileHandler(output_file, 'w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def collect_cuts_weights(selections):
    weights = []
    cuts = []
    for s in selections:
        if hasattr(s, 'weights'):
            t = []
            for w in s.weights:
                t += [w.expression]
            weights += t
        if hasattr(s, 'cuts'):
            t = []
            for c in s.cuts:
                t += [c.expression]
            cuts += t
    cutstr = '&&'.join(['({})'.format(c) for c in cuts])
    weightstr = '*'.join(['({})'.format(w) for w in weights])
    if cutstr == '':
        cutstr = 'true'
    if weightstr == '':
        weightstr = '1'
    return cutstr, weightstr


def make_chain(files, basepath):
    c = ROOT.TChain('mt_nominal/ntuple')
    for f in files:
        path = os.path.join(basepath, f, f + '.root')
        if not os.path.exists(path):
            logger.fatal('File %s does not exist', path)
            raise Exception
        c.AddFile(path)
    chains.append(c)
    return c


def make_dataset(files, ntuples_base, friends_base):
    n = make_chain(files, ntuples_base)
    for f in friends_base:
        c = make_chain(files, f)
        n.AddFriend(c)
    return n


def write_dataset(d, workdir, name, group, fold, weightstr, cutstr):
    df = ROOT.RDataFrame(d)
    variables = ROOT.std.vector(ROOT.std.string)()
    for v in cfg.ml_variables:
        variables.push_back(v)
    variables.push_back(cfg.ml_weight)
    df.Filter('event % 2 == {}'.format(fold))\
      .Filter(cutstr)\
      .Define(cfg.ml_weight, weightstr)\
      .Snapshot(group, os.path.join(workdir, '{}_fold{}.root'.format(name, fold)), variables)


def ggh():
    return cfg.files['ggh'], [cfg.channel, cfg.mc, cfg.htt, cfg.ggh], 'ggh', 'htt'


def qqh():
    return cfg.files['qqh'], [cfg.channel, cfg.mc, cfg.htt, cfg.qqh], 'qqh', 'htt'


def ztt():
    return cfg.files['dy'], [cfg.channel, cfg.mc, cfg.dy, cfg.ztt], 'ztt', 'ztt'


def w():
    return cfg.files['wjets'], [cfg.channel, cfg.mc, cfg.w], 'w', 'w'


def main(args):
    ROOT.EnableImplicitMT(args.nthreads)
    for process in [ggh, qqh, ztt, w]:
        files, selections, name, group = process()
        cutstr, weightstr = collect_cuts_weights(selections)
        d = make_dataset(files, cfg.ntuples_base, cfg.friends_base)
        logger.info('Create dataset for %s with label %s with %u events', process, name, d.GetEntries())
        logger.debug('Weight string: %s', weightstr)
        logger.debug('Cut string: %s', cutstr)
        for fold in [0, 1]:
            write_dataset(d, args.workdir, name, group, fold, weightstr, cutstr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    parser.add_argument('--nthreads', default=12, help='Number of threads')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'ml_dataset.log'), logging.INFO)
    main(args)