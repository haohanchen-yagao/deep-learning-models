from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow
from datetime import datetime
import os
import argparse
import importlib

now = datetime.now()
time_str = now.strftime("%d-%m-%Y-%H-%M")
date_str = now.strftime("%d-%m-%Y")

def main(args):
    loader = importlib.machinery.SourceFileLoader('', args.configuration)
    cfg = loader.load_module()
    #role = get_execution_role()
    role='arn:aws:iam::578276202366:role/service-role/AmazonSageMaker-ExecutionRole-20191213T102663'
    main_script = 'tools/train.py'
    docker_image = cfg.sagemaker_user['docker_image']
    #hvd_instance_count = cfg.sagemaker_user['hvd_instance_count']
    hvd_instance_count = args.instance_count
    #hvd_instance_type = cfg.sagemaker_user['hvd_instance_type']
    hvd_instance_type = args.instance_type
    distributions = cfg.distributions
    output_path = cfg.sagemaker_job['output_path']
    #job_name = cfg.sagemaker_job['job_name']
    job_name='chehaoha-{}rcnn-{}-nodes-{}'.format(args.configuration[8], hvd_instance_count, time_str)
    channels = cfg.channels

    configuration = {
        'config': args.configuration,
        'amp': 'True',
        'autoscale-lr': 'True',
        'validate': 'True'
    }

    estimator = TensorFlow(
                    entry_point=main_script, 
                    source_dir='.',
                    image_name=docker_image, 
                    role=role,
                    #framework_version="2.1.0",
                    framework_version="2.2.0",
                    py_version="py3",
                    train_instance_count=hvd_instance_count,
                    train_instance_type=hvd_instance_type,
                    distributions=distributions,
                    output_path=output_path,
                    train_volume_size=200,
                    hyperparameters=configuration)

    estimator.fit(channels, wait=True, job_name=job_name)
    print("Launched SageMaker job:", job_name)

def parse():
    """
    Parse path to configuration file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", help="SM Job configuration file")
    parser.add_argument("--instance_count", type=int, default=2)
    parser.add_argument("--instance_type", default="ml.p3dn.24xlarge")
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse()
    main(args)
