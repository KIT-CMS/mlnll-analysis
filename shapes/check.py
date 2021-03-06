import os
import argparse

from job import job as jobfunc

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


def main(args):
    jobdir = os.path.join(args.workdir, 'shapes_jobs')
    arguments = open(os.path.join(jobdir, 'arguments.txt'), 'r').readlines()
    missing_jobs = []
    for i, job in enumerate([x.split(' ') for x in arguments]):
        jobid = job[0]
        outputdir = os.path.join(job[2].strip(), 'shapes_files').strip()
        if not os.path.isdir(outputdir.strip()):
            logger.fatal('Output directory %s does not exist', outputdir)
            raise Exception
        outputfile = os.path.join(outputdir, 'shapes_{}.root'.format(jobid)).strip()
        logger.debug('Check job %s with output file %s', jobid, outputfile)
        if not os.path.isfile(outputfile):
            missing_jobs.append(arguments[i])

    logger.warn('Found %s missing jobs', len(missing_jobs))
    for job in missing_jobs:
        logger.info('Missing job found: %s', job)
        jobargs = job.split(' ')
        workdir = jobargs[2].strip()
        jobid = int(jobargs[0].strip())
        logger.info('Running locally with args (%s, %s)', jobid, workdir)
        jobfunc(workdir, jobid, nthreads=12)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'shapes_check.log'), logging.INFO)
    main(args)
