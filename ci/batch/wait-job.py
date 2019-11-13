import argparse
from datetime import datetime
import sys
import time

import boto3
from botocore.compat import total_seconds

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--profile', help='profile name of aws account.', type=str,
                    default=None)
parser.add_argument('--job-id', help='job id to check status and wait.', type=str,
                    default=None)

args = parser.parse_args()

session = boto3.Session(profile_name=args.profile)
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


def getLogStream(logGroupName, jobName, jobId):
    response = cloudwatch.describe_log_streams(
        logGroupName=logGroupName,
        logStreamNamePrefix=jobName + '/' + jobId
    )
    logStreams = response['logStreams']
    if not logStreams:
        return ''
    else:
        return logStreams[0]['logStreamName']

def nowInMillis():
    endTime = long(total_seconds(datetime.utcnow() - datetime(1970, 1, 1))) * 1000
    return endTime


def main():
    spin = ['-', '/', '|', '\\', '-', '/', '|', '\\']
    logGroupName = '/aws/batch/job'

    jobId = args.job_id

    spinner = 0
    running = False
    startTime = 0

    while True:
        time.sleep(1)
        describeJobsResponse = batch.describe_jobs(jobs=[jobId])
        job = describeJobsResponse['jobs'][0]
        status, jobName = job['status'], job['jobName']
        if status == 'SUCCEEDED' or status == 'FAILED':
            print('=' * 80)
            print('Job [{} - {}] {}'.format(jobName, jobId, status))
            break
        elif status == 'RUNNING':
            logStreamName = getLogStream(logGroupName, jobName, jobId)
            if not running and logStreamName:
                running = True
                print('\rJob [{} - {}] is RUNNING.'.format(jobName, jobId))
                print('Output [{}]:\n {}'.format(logStreamName, '=' * 80))
            if logStreamName:
                startTime = printLogs(logGroupName, logStreamName, startTime) + 1
        else:
            print('\rJob [%s - %s] is %-9s... %s' % (jobName, jobId, status, spin[spinner % len(spin)]),)
            sys.stdout.flush()
            spinner += 1

if __name__ == '__main__':
    main()
