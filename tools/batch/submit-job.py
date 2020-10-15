import argparse
import random
import re
import sys
import time
from datetime import datetime

import boto3
from botocore.compat import total_seconds

instance_type_info = {
    'g4dn.4x': {
        'job_definition': 'gluon-nlp-g4dn_4xlarge:5',
        'job_queue': 'g4dn'
    },
    'g4dn.8x': {
        'job_definition': 'gluon-nlp-g4dn_8xlarge:5',
        'job_queue': 'g4dn'
    },
    'g4dn.12x': {
        'job_definition': 'gluon-nlp-g4dn_12xlarge:5',
        'job_queue': 'g4dn-multi-gpu'
    },
    'p3.2x': {
        'job_definition': 'gluon-nlp-p3_2xlarge:5',
        'job_queue': 'p3'
    },
    'p3.8x': {
        'job_definition': 'gluon-nlp-p3_8xlarge:5',
        'job_queue': 'p3-4gpu'
    },
    'p3.16x': {
        'job_definition': 'gluon-nlp-p3_16xlarge:5',
        'job_queue': 'p3-8gpu'
    },
    'p3dn.24x': {
        'job_definition': 'gluon-nlp-p3_24xlarge:5',
        'job_queue': 'p3dn-8gpu'
    },
    'c5n.4x': {
        'job_definition': 'gluon-nlp-c5_4xlarge:3',
        'job_queue': 'c5n'
    },
    'c5n.18x': {
        'job_definition': 'gluon-nlp-c5_18xlarge:3',
        'job_queue': 'c5n'
    }
}

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--profile', help='profile name of aws account.', type=str,
                    default=None)
parser.add_argument('--region', help='Default region when creating new connections', type=str,
                    default=None)
parser.add_argument('--name', help='name of the job', type=str, default='dummy')
parser.add_argument('--job-type', help='type of job to submit.', type=str,
                    choices=instance_type_info.keys(), default='g4dn.4x')
parser.add_argument('--source-ref',
                    help='ref in GluonNLP main github. e.g. master, refs/pull/500/head',
                    type=str, default='master')
parser.add_argument('--work-dir',
                    help='working directory inside the repo. e.g. scripts/preprocess',
                    type=str, default='scripts/preprocess')
parser.add_argument('--saved-output',
                    help='output to be saved, relative to working directory. '
                         'it can be either a single file or a directory',
                    type=str, default='.')
parser.add_argument('--save-path',
                    help='s3 path where files are saved.',
                    type=str, default='batch/temp/{}'.format(datetime.now().isoformat()))
parser.add_argument('--command', help='command to run', type=str,
                    default='git rev-parse HEAD | tee stdout.log')
parser.add_argument('--remote',
                    help='git repo address. https://github.com/dmlc/gluon-nlp',
                    type=str, default="https://github.com/dmlc/gluon-nlp")
parser.add_argument('--wait', help='block wait until the job completes. '
                    'Non-zero exit code if job fails.', action='store_true')
parser.add_argument('--timeout', help='job timeout in seconds', default=None, type=int)


args = parser.parse_args()

session = boto3.Session(profile_name=args.profile, region_name=args.region)
batch, cloudwatch = [session.client(service_name=sn) for sn in ['batch', 'logs']]


def printLogs(logGroupName, logStreamName, startTime):
    kwargs = {'logGroupName': logGroupName,
              'logStreamName': logStreamName,
              'startTime': startTime,
              'startFromHead': True}

    lastTimestamp = 0
    while True:
        logEvents = cloudwatch.get_log_events(**kwargs)

        for event in logEvents['events']:
            lastTimestamp = event['timestamp']
            timestamp = datetime.utcfromtimestamp(lastTimestamp / 1000.0).isoformat()
            print('[{}] {}'.format((timestamp + '.000')[:23] + 'Z', event['message']))

        nextToken = logEvents['nextForwardToken']
        if nextToken and kwargs.get('nextToken') != nextToken:
            kwargs['nextToken'] = nextToken
        else:
            break
    return lastTimestamp


def nowInMillis():
    endTime = long(total_seconds(datetime.utcnow() - datetime(1970, 1, 1))) * 1000
    return endTime


def main():
    spin = ['-', '/', '|', '\\', '-', '/', '|', '\\']
    logGroupName = '/aws/batch/job'

    jobName = re.sub('[^A-Za-z0-9_\-]', '', args.name)[:128]  # Enforce AWS Batch jobName rules
    jobType = args.job_type
    jobQueue = instance_type_info[jobType]['job_queue']
    jobDefinition = instance_type_info[jobType]['job_definition']
    command = args.command.split()
    wait = args.wait

    parameters = {
        'SOURCE_REF': args.source_ref,
        'WORK_DIR': args.work_dir,
        'SAVED_OUTPUT': args.saved_output,
        'SAVE_PATH': args.save_path,
        'COMMAND': args.command,
        'REMOTE': args.remote
    }
    kwargs = dict(
        jobName=jobName,
        jobQueue=jobQueue,
        jobDefinition=jobDefinition,
        parameters=parameters,
    )
    if args.timeout is not None:
        kwargs['timeout'] = {'attemptDurationSeconds': args.timeout}
    submitJobResponse = batch.submit_job(**kwargs)

    jobId = submitJobResponse['jobId']
    print('Submitted job [{} - {}] to the job queue [{}]'.format(jobName, jobId, jobQueue))

    spinner = 0
    running = False
    status_set = set()
    startTime = 0
    logStreamName = None
    while wait:
        time.sleep(random.randint(5, 10))
        describeJobsResponse = batch.describe_jobs(jobs=[jobId])
        status = describeJobsResponse['jobs'][0]['status']
        if status == 'SUCCEEDED' or status == 'FAILED':
            print('=' * 80)
            print('Job [{} - {}] {}'.format(jobName, jobId, status))
            if logStreamName:
                startTime = printLogs(logGroupName, logStreamName, startTime) + 1
            sys.exit(status == 'FAILED')

        elif status == 'RUNNING':
            logStreamName = describeJobsResponse['jobs'][0]['container']['logStreamName']
            if not running:
                running = True
                print('\rJob [{}, {}] is RUNNING.'.format(jobName, jobId))
                if logStreamName:
                    print('Output [{}]:\n {}'.format(logStreamName, '=' * 80))
            if logStreamName:
                startTime = printLogs(logGroupName, logStreamName, startTime) + 1
        elif status not in status_set:
            status_set.add(status)
            print('\rJob [%s - %s] is %-9s... %s' % (jobName, jobId, status, spin[spinner % len(spin)]),)
            sys.stdout.flush()
            spinner += 1


if __name__ == '__main__':
    main()
