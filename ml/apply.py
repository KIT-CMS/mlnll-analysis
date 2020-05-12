import os
import argparse

import ROOT
import numpy as np
np.random.seed(1234)
from sklearn.metrics import confusion_matrix
from utils import config as cfg
import tensorflow as tf
tf.set_random_seed(1234)

from array import array

from collect_events import make_dataset

import logging
logger = logging.getLogger('')


def setup_logging(output_file, level=logging.DEBUG):
    logger.setLevel(level)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    file_handler = logging.FileHandler(output_file, 'w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


foldernames = [
        'mt_nominal',
        'mt_jerUncUp',
        'mt_jerUncDown',
        'mt_jecUncAbsoluteYearUp',
        'mt_jecUncAbsoluteYearDown',
        'mt_jecUncAbsoluteUp',
        'mt_jecUncAbsoluteDown',
        'mt_jecUncBBEC1YearUp',
        'mt_jecUncBBEC1YearDown',
        'mt_jecUncBBEC1Up',
        'mt_jecUncBBEC1Down',
        'mt_jecUncEC2YearUp',
        'mt_jecUncEC2YearDown',
        'mt_jecUncEC2Up',
        'mt_jecUncEC2Down',
        'mt_jecUncHFYearUp',
        'mt_jecUncHFYearDown',
        'mt_jecUncHFUp',
        'mt_jecUncHFDown',
        'mt_jecUncFlavorQCDUp',
        'mt_jecUncFlavorQCDDown',
        'mt_jecUncRelativeSampleYearUp',
        'mt_jecUncRelativeSampleYearDown',
        'mt_jecUncRelativeBalUp',
        'mt_jecUncRelativeBalDown',
        'mt_tauEsThreeProngUp',
        'mt_tauEsThreeProngDown',
        'mt_tauEsThreeProngOnePiZeroUp',
        'mt_tauEsThreeProngOnePiZeroDown',
        'mt_tauEsOneProngUp',
        'mt_tauEsOneProngDown',
        'mt_tauEsOneProngOnePiZeroUp',
        'mt_tauEsOneProngOnePiZeroDown',
        ]


def main(args):
    model_fold0 = tf.keras.models.load_model(os.path.join(args.workdir, 'model_fold0.h5'))
    model_fold1 = tf.keras.models.load_model(os.path.join(args.workdir, 'model_fold1.h5'))

    for process in ['ggh']:#cfg.files:
        logger.info('Process files of process {}'.format(process))
        for filename in cfg.files[process]:
            for folder in foldernames:
                # Check whether the input file and folder exist
                # Just skip over missing folders but break for missing files
                filepath = os.path.join(cfg.ntuples_base, filename, filename + '.root')
                folderpath = folder + '/ntuple'
                logger.info('Process folder {} of file {}'.format(folderpath, filepath))
                f = ROOT.TFile(filepath, 'READ')
                if f == None:
                    logger.fatal('File {} does not exist'.format(filepath))
                    raise Exception

                dir_ = f.Get(folderpath)
                if dir_ == None:
                    logger.warn('Skipping over folder {} in file {} (does not exist)'.format(folderpath, filepath))
                    f.Close()
                    continue
                f.Close()

                # Create chain with friends
                d = make_dataset([filename], cfg.ntuples_base, cfg.friends_base, folder)
                num_entries = d.GetEntries()
                logger.debug('Process file {} with {} events'.format(filename, num_entries))

                # Convert to numpy and stack to input dataset
                npy = ROOT.RDataFrame(d).AsNumpy(cfg.ml_variables + ['event'])
                inputs  = np.vstack([np.array(npy[k], dtype=np.float32) for k in cfg.ml_variables]).T

                # Apply model of fold 0 to data of fold 1 and v.v.
                mask_fold0 = npy['event'] % 2 == 0
                mask_fold1 = npy['event'] % 2 == 1
                logger.debug('Events in fold 0 / fold 1 / total: {} / {} / {}'.format(np.sum(mask_fold0), np.sum(mask_fold1), num_entries))
                if np.sum(mask_fold0) + np.sum(mask_fold1) != num_entries:
                    logger.fatal('Events in folds dont add up to expected total')
                    raise Exception

                outputs_fold0 = model_fold1.predict(inputs[mask_fold0])
                scores_fold0 = np.max(outputs_fold0, axis=1)
                indices_fold0 = np.argmax(outputs_fold0, axis=1)

                outputs_fold1 = model_fold0.predict(inputs[mask_fold1])
                scores_fold1 = np.max(outputs_fold1, axis=1)
                indices_fold1 = np.argmax(outputs_fold1, axis=1)

                # Merge scores back together
                scores = np.zeros(npy['event'].shape, dtype=np.float32)
                scores[mask_fold0] = scores_fold0
                scores[mask_fold1] = scores_fold1

                indices = np.zeros(npy['event'].shape, dtype=np.float32)
                indices[mask_fold0] = indices_fold0
                indices[mask_fold1] = indices_fold1

                # Try to make output folder
                try:
                    os.mkdir(os.path.join(args.workdir, 'MLScores', filename))
                except:
                    pass

                # Write output file
                f = ROOT.TFile(os.path.join(args.workdir, 'MLScores', filename, filename + '.root'), 'UPDATE')
                dir_ = f.mkdir(folder)
                dir_.cd()
                t = ROOT.TTree('ntuple', 'ntuple')
                val = array('f', [-999])
                idx = array('f', [-999])
                bval = t.Branch('ml_score', val, 'ml_score/F')
                bidx = t.Branch('ml_index', idx, 'ml_index/F')
                for i in range(scores.shape[0]):
                    val[0] = scores[i]
                    idx[0] = indices[i]
                    t.Fill()
                t.Write()
                f.Close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'ml_apply.log'), logging.INFO)
    main(args)
