import os, sys
import boto3
import re
import argparse
from datetime import datetime

folder='NLP-HERRING'

def regex_extract(text, pattern):
    m = re.search(pattern, text)
    print(text)
    if m:
        found = m.group(1)
    return found

def extract_result(log_abspath):
    result = {}
    loss = 100
    mlm = 0
    sop = 0
    throughput = 0
    started = False
    with open(log_abspath, 'r') as log:
        for line in log:
            if 'Validation' in line and 'MLM_acc: ' in line:
                temp_mlm = float(regex_extract(line, 'MLM_acc: ([-+]?\d*\.\d+|\d+)'))
                mlm = temp_mlm
            if 'Validation' in line and 'SOP_acc: ' in line:
                temp_sop = float(regex_extract(line, 'SOP_acc: ([-+]?\d*\.\d+|\d+)'))
                sop = temp_sop
            '''if 'Validation' in line and 'Loss: ' in line and 'MLM: ' in line:
                temp_loss = float(regex_extract(line, 'Loss: ([-+]?\d*\.\d+|\d+)'))
                loss = temp_loss'''
            if 'W tensorflow' in line and started==False:
                start_string = regex_extract(line, '(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?=.))')
                start = datetime.strptime(start_string, '%Y-%m-%d %H:%M:%S')
                started=True
            if 'Finished pretraining' in line:
                end_string = regex_extract(line, '(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?=,))')
                end = datetime.strptime(end_string, '%Y-%m-%d %H:%M:%S')
            if 'It/s: ' in line:
                throughput = float(regex_extract(line, 'It/s: ([-+]?\d*\.\d+|\d+)'))
                result['throughput'] = throughput
    result['mlm'] = mlm
    result['sop'] = sop
    result['loss'] = loss
    print(start)
    print(end)
    diff = end - start
    print(diff.total_seconds())
    result['time'] = diff.total_seconds()
    return result

def upload_metrics(parsed_results, num_gpus, batch_size, instance_type, platform, trigger, model):
    client = boto3.client('cloudwatch')
    '''print(parsed_results['loss'])
    client.put_metric_data(
      Namespace='{}_{}'.format(folder, model),
      MetricData=[
        {
          'MetricName': 'Loss',
          'Value': parsed_results['loss'],
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
    )'''
    print(parsed_results['mlm'])
    client.put_metric_data(
      Namespace='{}_{}'.format(folder, model),
      MetricData=[
        {
          'MetricName': 'MLM Accuracy',
          'Value': parsed_results['mlm'],
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
    print(parsed_results['sop'])
    client.put_metric_data(
      Namespace='{}_{}'.format(folder, model),
      MetricData=[
        {
          'MetricName': 'SOP Accuracy',
          'Value': parsed_results['sop'],
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
    print("throughput is {}it/s".format(parsed_results['throughput']))
    client.put_metric_data(
      Namespace='{}_{}'.format(folder, model),
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
    print("Duration is {} sec, which is {} min, {} h".format(parsed_results['time'], parsed_results['time']/60, parsed_results['time']/3600))
    client.put_metric_data(
      Namespace='{}_{}'.format(folder, model),
      MetricData=[
        {
          'MetricName': 'Training time',
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
    parser.add_argument('--model', type=str, default='ALBERT')
    args = parser.parse_args()
    abspath = os.path.join(os.getcwd(), args.log)
    parsed_results = extract_result(abspath)
    print(parsed_results['throughput'])
    print(parsed_results['loss'])
   
    if parsed_results['mlm'] > 0:
        upload_metrics(parsed_results, args.num_gpus, args.batchsize, args.instance_type, args.platform, args.trigger, args.model)
    else:
        print('nothing being parsed')