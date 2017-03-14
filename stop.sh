#!/bin/bash

set -e
set -x

#instanceid='i-0c945b5460b439785'
instanceid='i-0781898f584706a86'

echo "Stopping instance"
aws ec2 stop-instances --instance-ids "${instanceid}"
