#!/usr/bin/env python3

import boto3
import time

ec2 = boto3.client('ec2')

response = ec2.request_spot_instances(
    SpotPrice="0.10",
    InstanceCount=1,
    Type="one-time",
    LaunchSpecification={
        "ImageId": "ami-e369c4f5",
        "InstanceType": "g2.2xlarge",
        #"InstanceType": "c3.large",
        "KeyName": "carnd",
        "SecurityGroupIds": [
            "sg-59e96a4f"
        ],
        #"SecurityGroups": [
        #],
        "UserData": "",
        "Placement": {
            "AvailabilityZone": "us-east-1c"
        },
        "IamInstanceProfile": {
            "Arn": "arn:aws:iam::205765788265:instance-profile/carnd"
        }
    }
)

#print(response)

spot_request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
print("Spot request id={}".format(spot_request_id))

def wait_for_instance(spot_request_id):
    instance_id = None

    for i in range(40):
        response = ec2.describe_spot_instance_requests(SpotInstanceRequestIds=[spot_request_id])
        state = response['SpotInstanceRequests'][0]['State']
        # state in ('open', 'closed', 'failed', 'cancelled', 'active')
        if state in ('closed', 'failed', 'cancelled'):
            return None
        if state in ('active', ):
            instance_id = response['SpotInstanceRequests'][0]['InstanceId']
            break
        time.sleep(15)

    return instance_id

instance_id = wait_for_instance(spot_request_id)
response = ec2.cancel_spot_instance_requests(SpotInstanceRequestIds=[spot_request_id])

if instance_id is None:
    print("Was not able to allocate instance")
else:
    print("Launched instance: {}".format(instance_id))
    inst = boto3.resource('ec2').Instance(instance_id)
    inst.wait_until_running()
    print("Instance public DNS: {}", inst.public_dns_name)
    print("Instance should running, status: {}".format(inst.state))
