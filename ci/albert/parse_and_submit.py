import os, sys
import boto3
import re
import argparse

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
    with open(log_abspath, 'r') as log:
        for line in log:
            if 'MLM_acc: ' in line:
                temp_mlm = float(regex_extract(line, 'MLM_acc: ([-+]?\d*\.\d+|\d+)'))
                mlm = max(mlm, temp_mlm)
            if 'SOP_acc: ' in line:
                temp_sop = float(regex_extract(line, 'SOP_acc: ([-+]?\d*\.\d+|\d+)'))
                sop = max(sop, temp_sop)
            if 'Loss: ' in line and 'MLM: ' in line:
                temp_loss = float(regex_extract(line, 'Loss: ([-+]?\d*\.\d+|\d+)'))
                loss = min(loss, temp_loss)
            if 'Training seconds: ' in line:
                time = float(regex_extract(line, 'Training seconds: ([-+]?\d*\.\d+|\d+)'))
                result['time'] = time
            if 'EM: ' in line and 'it/s' in line:
                throughput = float(regex_extract(line, '([-+]?\d*\.\d+|\d+)it/s'))
                result['throughput'] = throughput
    result['mlm'] = mlm
    result['sop'] = sop
    result['loss'] = loss
    return result

def upload_metrics(parsed_results, num_gpus, batch_size, instance_type, platform, trigger):
    client = boto3.client('cloudwatch')
    print(parsed_results['loss'])
    client.put_metric_data(
      Namespace='ModelOptimization',
      MetricData=[
        {
          'MetricName': 'Loss',
          'Value': parsed_results['loss'],
          'Dimensions': [
              {
                  'Name': 'Model',
                  'Value': 'ALBERT'
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
    print(parsed_results['mlm'])
    client.put_metric_data(
      Namespace='ModelOptimization',
      MetricData=[
        {
          'MetricName': 'MLM Accuracy',
          'Value': parsed_results['mlm'],
          'Dimensions': [
              {
                  'Name': 'Model',
                  'Value': 'ALBERT'
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
      Namespace='ModelOptimization',
      MetricData=[
        {
          'MetricName': 'SOP Accuracy',
          'Value': parsed_results['sop'],
          'Dimensions': [
              {
                  'Name': 'Model',
                  'Value': 'ALBERT'
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
      Namespace='ModelOptimization',
      MetricData=[
        {
          'MetricName': 'Training time',
          'Value': parsed_results['sop'],
          'Dimensions': [
              {
                  'Name': 'Model',
                  'Value': 'ALBERT'
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
    args = parser.parse_args()
    abspath = os.path.join(os.getcwd(), args.log)
    parsed_results = extract_result(abspath)
    print(parsed_results['throughput'])
    print(parsed_results['loss'])
   
    if parsed_results['loss'] < 100:
        upload_metrics(parsed_results, args.num_gpus, args.batchsize, args.instance_type, args.platform, args.trigger)
    else:
        print('nothing being parsed')