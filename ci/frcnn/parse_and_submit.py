import os, sys
import boto3
import re
import argparse
from datetime import datetime

folder = "Models_CV"

def regex_extract(text, pattern):
    m = re.search(pattern, text)
    print(text)
    if m:
        found = m.group(1)
    return found

def extract_result(log_abspath):
    result = {}
    ap_95_all = 0
    ap_50_all = 0
    ap_75_all = 0
    ap_95_small = 0
    ap_95_medium = 0
    ap_95_large = 0
    start = datetime.now()
    end = datetime.now()
    with open(log_abspath, 'r') as log:
        for line in log:
            if 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100' in line:
                temp_ap_95_all = float(regex_extract(line, '(?<=maxDets\=100\s\]\s\=\s)([-+]?\d*\.\d+|\d+)'))
                ap_95_all = max(ap_95_all, temp_ap_95_all)
            if 'Average Precision  (AP) @[ IoU=0.50 ' in line:
                temp_ap_50_all = float(regex_extract(line, '(?<=maxDets\=100\s\]\s\=\s)([-+]?\d*\.\d+|\d+)'))
                ap_50_all = max(ap_50_all, temp_ap_50_all)
            if 'Average Precision  (AP) @[ IoU=0.75 ' in line:
                temp_ap_75_all = float(regex_extract(line, '(?<=maxDets\=100\s\]\s\=\s)([-+]?\d*\.\d+|\d+)'))
                ap_75_all = max(ap_75_all, temp_ap_75_all)
            if 'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100' in line:
                temp_ap_95_small = float(regex_extract(line, '(?<=maxDets\=100\s\]\s\=\s)([-+]?\d*\.\d+|\d+)'))
                ap_95_small = max(ap_95_small, temp_ap_95_small)
            if 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100' in line:
                temp_ap_95_medium = float(regex_extract(line, '(?<=maxDets\=100\s\]\s\=\s)([-+]?\d*\.\d+|\d+)'))
                ap_95_medium = max(ap_95_medium, temp_ap_95_medium)
            if 'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100' in line:
                temp_ap_95_large = float(regex_extract(line, '(?<=maxDets\=100\s\]\s\=\s)([-+]?\d*\.\d+|\d+)'))
                ap_95_large = max(ap_95_large, temp_ap_95_large)
            if 'Training image download completed. Training in progress' in line:
                start_string = regex_extract(line, '(.+?(?= Training - Training))')
                start = datetime.strptime(start_string, '%Y-%m-%d %H:%M:%S')
            if 'MPI process finished' in line:
                end_string = regex_extract(line, '(.+?(?=,))')
                end = datetime.strptime(end_string, '%Y-%m-%d %H:%M:%S')
            if 'task/s' in line:
                throughput = float(regex_extract(line, '([-+]?\d*\.\d+|\d+) task/s'))
                result['throughput'] = throughput
    result['ap_0.95_all'] = ap_95_all
    result['ap_0.50_all'] = ap_50_all
    result['ap_0.75_all'] = ap_75_all
    result['ap_0.95_small'] = ap_95_small
    result['ap_0.95_medium'] = ap_95_medium
    result['ap_0.95_large'] = ap_95_large
    diff = end - start
    result['time'] = diff.seconds
    return result

def upload_metrics(parsed_results, num_gpus, batch_size, instance_type, platform, trigger, model):
    client = boto3.client('cloudwatch')
    print(parsed_results['ap_0.95_all'])
    client.put_metric_data(
      Namespace=folder,
      MetricData=[
        {
          'MetricName': 'Precision-0.50:0.95-all',
          'Value': parsed_results['ap_0.95_all'],
          'Dimensions': [
              {
                  'Name': 'Model',
                  'Value': model
              },
              {
                  'Name': 'Platform',
                  'Value': str(platform)
              },
              {
                  'Name': 'Instance Type',
                  'Value': str(instance_type)
              },
              {
                  'Name': 'Num of GPUs',
                  'Value': 'GPUs:' + str(num_gpus)
              },
              {
                  'Name': 'Batch Size',
                  'Value': 'Batche Size:' + str(batch_size)
              },
              {
                  'Name': 'Trigger',
                  'Value': str(trigger)
              }
          ]
        }
      ]
    )
    print(parsed_results['ap_0.50_all'])
    client.put_metric_data(
      Namespace=folder,
      MetricData=[
        {
          'MetricName': 'Precision-0.50-all',
          'Value': parsed_results['ap_0.50_all'],
          'Dimensions': [
              {
                  'Name': 'Model',
                  'Value': model
              },
              {
                  'Name': 'Platform',
                  'Value': str(platform)
              },
              {
                  'Name': 'Instance Type',
                  'Value': str(instance_type)
              },
              {
                  'Name': 'Num of GPUs',
                  'Value': 'GPUs:' + str(num_gpus)
              },
              {
                  'Name': 'Batch Size',
                  'Value': 'Batch Size:' + str(batch_size)
              },
              {
                  'Name': 'Trigger',
                  'Value': str(trigger)
              }
          ]
        }
      ]
    )
    print(parsed_results['ap_0.75_all'])
    client.put_metric_data(
      Namespace=folder,
      MetricData=[
        {
          'MetricName': 'Precision-0.75-all',
          'Value': parsed_results['ap_0.75_all'],
          'Dimensions': [
              {
                  'Name': 'Model',
                  'Value': model
              },
              {
                  'Name': 'Platform',
                  'Value': str(platform)
              },
              {
                  'Name': 'Instance Type',
                  'Value': str(instance_type)
              },
              {
                  'Name': 'Num of GPUs',
                  'Value': 'GPUs:' + str(num_gpus)
              },
              {
                  'Name': 'Batch Size',
                  'Value': 'Batch Size:' + str(batch_size)
              },
              {
                  'Name': 'Trigger',
                  'Value': str(trigger)
              }
          ]
        }
      ]
    )
    print(parsed_results['ap_0.95_small'])
    client.put_metric_data(
      Namespace=folder,
      MetricData=[
        {
          'MetricName': 'Precision-0.50:0.95-small',
          'Value': parsed_results['ap_0.95_small'],
          'Dimensions': [
              {
                  'Name': 'Model',
                  'Value': model
              },
              {
                  'Name': 'Platform',
                  'Value': str(platform)
              },
              {
                  'Name': 'Instance Type',
                  'Value': str(instance_type)
              },
              {
                  'Name': 'Num of GPUs',
                  'Value': 'GPUs:' + str(num_gpus)
              },
              {
                  'Name': 'Batch Size',
                  'Value': 'Batch Size:' + str(batch_size)
              },
              {
                  'Name': 'Trigger',
                  'Value': str(trigger)
              }
          ]
        }
      ]
    )
    print(parsed_results['ap_0.95_medium'])
    client.put_metric_data(
      Namespace=folder,
      MetricData=[
        {
          'MetricName': 'Precision-0.50:0.95-medium',
          'Value': parsed_results['ap_0.95_medium'],
          'Dimensions': [
              {
                  'Name': 'Model',
                  'Value': model
              },
              {
                  'Name': 'Platform',
                  'Value': str(platform)
              },
              {
                  'Name': 'Instance Type',
                  'Value': str(instance_type)
              },
              {
                  'Name': 'Num of GPUs',
                  'Value': 'GPUs:' + str(num_gpus)
              },
              {
                  'Name': 'Batch Size',
                  'Value': 'Batch Size:' + str(batch_size)
              },
              {
                  'Name': 'Trigger',
                  'Value': str(trigger)
              }
          ]
        }
      ]
    )
    print(parsed_results['ap_0.95_large'])
    client.put_metric_data(
      Namespace=folder,
      MetricData=[
        {
          'MetricName': 'Precision-0.50:0.95-large',
          'Value': parsed_results['ap_0.95_large'],
          'Dimensions': [
              {
                  'Name': 'Model',
                  'Value': model
              },
              {
                  'Name': 'Platform',
                  'Value': str(platform)
              },
              {
                  'Name': 'Instance Type',
                  'Value': str(instance_type)
              },
              {
                  'Name': 'Num of GPUs',
                  'Value': 'GPUs:' + str(num_gpus)
              },
              {
                  'Name': 'Batch Size',
                  'Value': 'Batch Size:' + str(batch_size)
              },
              {
                  'Name': 'Trigger',
                  'Value': str(trigger)
              }
          ]
        }
      ]
    )
    print(parsed_results['throughput'])
    client.put_metric_data(
      Namespace=folder,
      MetricData=[
        {
          'MetricName': 'Throughput',
          'Value': parsed_results['throughput'],
          'Dimensions': [
              {
                  'Name': 'Model',
                  'Value': model
              },
              {
                  'Name': 'Platform',
                  'Value': str(platform)
              },
              {
                  'Name': 'Instance Type',
                  'Value': str(instance_type)
              },
              {
                  'Name': 'Num of GPUs',
                  'Value': 'GPUs:' + str(num_gpus)
              },
              {
                  'Name': 'Batch Size',
                  'Value': 'Batch Size:' + str(batch_size)
              },
              {
                  'Name': 'Trigger',
                  'Value': str(trigger)
              }
          ]
        }
      ]
    )
    print(parsed_results['time'])
    client.put_metric_data(
      Namespace=folder,
      MetricData=[
        {
          'MetricName': 'Duration',
          'Value': parsed_results['time'],
          'Dimensions': [
              {
                  'Name': 'Model',
                  'Value': model
              },
              {
                  'Name': 'Platform',
                  'Value': str(platform)
              },
              {
                  'Name': 'Instance Type',
                  'Value': str(instance_type)
              },
              {
                  'Name': 'Num of GPUs',
                  'Value': 'GPUs:' + str(num_gpus)
              },
              {
                  'Name': 'Batch Size',
                  'Value': 'Batch Size:' + str(batch_size)
              },
              {
                  'Name': 'Trigger',
                  'Value': str(trigger)
              }
          ]
        }
      ]
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='log.txt')
    parser.add_argument('--num_gpus', type=str, default='8')
    parser.add_argument('--batchsize', type=str, default='32')
    parser.add_argument('--instance_type', type=str, default='ml.p3dn.24xlarge')
    parser.add_argument('--platform', type=str, default='Sagemaker')
    parser.add_argument('--trigger', type=str, default='Weekly')
    parser.add_argument('--model', type=str, default='FasterRCNN')
    args = parser.parse_args()
    abspath = os.path.join(os.getcwd(), args.log)
    parsed_results = extract_result(abspath)
    print(parsed_results['throughput'])
    if parsed_results['time'] > 0:
        upload_metrics(parsed_results, args.num_gpus, args.batchsize, args.instance_type, args.platform, args.trigger, args.model)
    else:
        print('nothing being parsed')