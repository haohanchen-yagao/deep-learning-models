import os, sys
import boto3
import re
import argparse
from datetime import datetime

space = "cv-herring"

def regex_extract(text, pattern):
    m = re.search(pattern, text)
    print(text)
    if m:
        found = m.group(1)
    return found

def extract_result(log_abspath, epochs, gpus, batchsize):
    result = {}
    bbox = 0
    segm = 0
    time = 0
    data_time = 0
    sample_time = 0
    sample_data_time = 0
    time_count = 0
    eval_time = 0
    iterations = 0
    throughput = 0
    avg_time = 0 
    avg_data_time = 0
    started = False
    with open(log_abspath, 'r') as log:
        for line in log:
            if 'bbox_mAP_copypaste' in line:
                bbox = float(regex_extract(line, '(?<=bbox\_mAP\_copypaste\:\s)([-+]?\d*\.\d+|\d+)'))
            if 'segm_mAP_copypaste' in line:
                segm = float(regex_extract(line, '(?<=segm\_mAP\_copypaste\:\s)([-+]?\d*\.\d+|\d+)'))
            if 'Epoch' in line and 'data_time' in line:
                iterations = float(regex_extract(line, '((?<=\/)[0-9]+(?=\]))'))
                print(iterations)
                sample_time += float(regex_extract(line, '(?<=\,\stime:\s)([-+]?\d*\.\d+|\d+)'))
                sample_data_time += float(regex_extract(line, '(?<=\,\sdata\_time:\s)([-+]?\d*\.\d+|\d+)'))
                time_count += 1
            if 'eval_time' in line:
                eval_time += float(regex_extract(line, '(?<=eval\_time:\s)([-+]?\d*\.\d+|\d+)'))
            if 'Tensorflow version' in line and started==False:
                start_string = regex_extract(line, '(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?=,))')
                start = datetime.strptime(start_string, '%Y-%m-%d %H:%M:%S')
                started=True
    if time_count != 0:
        avg_time = sample_time / time_count
        print("avg time is {}".format(avg_time))
        avg_data_time = sample_data_time / time_count
        throughput = (batchsize * gpus) / avg_time
        total_iteration = iterations * epochs
        time = avg_time * total_iteration
        data_time = avg_data_time * total_iteration
    result['bbox'] = bbox
    result['segm'] = segm
    result['time'] = time
    result['data_time'] = avg_data_time
    result['throughput'] = throughput
    result['eval_time'] = eval_time 
    result['time_per_it'] = avg_time
    return result

def upload_metrics(parsed_results, num_gpus, batch_size, instance_type, platform, trigger, model):
    folder = "{}_{}".format(space, model)
    client = boto3.client('cloudwatch')
    print("bbox precision is {}".format(parsed_results['bbox']))
    client.put_metric_data(
      Namespace=folder,
      MetricData=[
        {
          'MetricName': 'bbox-precision',
          'Value': parsed_results['bbox'],
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
    print("segm precision is {}".format(parsed_results['segm']))
    client.put_metric_data(
      Namespace=folder,
      MetricData=[
        {
          'MetricName': 'segm-precision',
          'Value': parsed_results['segm'],
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
    print("throughput is {} it/s".format(parsed_results['throughput']))
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
    print("Training time is {} sec, which is {} min".format(parsed_results['time'], parsed_results['time']/60))
    client.put_metric_data(
      Namespace=folder,
      MetricData=[
        {
          'MetricName': 'Training Time',
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
    print("Data time is {} sec".format(parsed_results['data_time']))
    client.put_metric_data(
      Namespace=folder,
      MetricData=[
        {
          'MetricName': 'Data Time per iteration',
          'Value': parsed_results['data_time'],
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
    print("Eval time is {} sec, which is {} min".format(parsed_results['eval_time'], parsed_results['eval_time']/60))
    client.put_metric_data(
      Namespace=folder,
      MetricData=[
        {
          'MetricName': 'Evaluation Time',
          'Value': parsed_results['eval_time'],
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
    print("Time per it is {} sec".format(parsed_results['time_per_it']))
    client.put_metric_data(
      Namespace=folder,
      MetricData=[
        {
          'MetricName': 'Time per it',
          'Value': parsed_results['time_per_it'],
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
    parser.add_argument('--num_gpus', type=int, default='8')
    parser.add_argument('--batchsize', type=int, default='4')
    parser.add_argument('--instance_type', type=str, default='ml.p3dn.24xlarge')
    parser.add_argument('--platform', type=str, default='Sagemaker')
    parser.add_argument('--trigger', type=str, default='Monthly')
    parser.add_argument('--model', type=str, default='fasterrcnn')
    parser.add_argument('--epochs', type=int, default=12)
    args = parser.parse_args()
    abspath = os.path.join(os.getcwd(), args.log)
    parsed_results = extract_result(abspath, args.epochs, args.num_gpus, args.batchsize)
    print(parsed_results['throughput'])
    if parsed_results['time'] > 0:
        upload_metrics(parsed_results, args.num_gpus, args.batchsize, args.instance_type, args.platform, args.trigger, args.model)
    else:
        print('nothing being parsed')