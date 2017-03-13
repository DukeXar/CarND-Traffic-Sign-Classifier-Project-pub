#!/bin/bash

set -e
set -x

hostedzoneid='/hostedzone/Z1225HXENZMLQ9'
#instanceid='i-0c945b5460b439785'
instanceid='i-0781898f584706a86'

state=$(aws ec2 describe-instances --instance-ids "${instanceid}" | jq '.Reservations[0].Instances[0].State.Name')

if [ "$state" != "\"running\"" ]; then
    echo "Starting instance"
    aws ec2 start-instances --instance-ids "${instanceid}"
else
    echo "Instance is running"
fi

echo "Retrieving instance IP address"
public_ip=$(aws ec2 describe-instances --instance-ids "${instanceid}" | jq '.Reservations[0].Instances[0].PublicIpAddress')

# aws route53 list-resource-record-sets --hosted-zone-id "${hostedzoneid}"

request=$(cat <<EOF
{
    "HostedZoneId": "${hostedzoneid}",
    "ChangeBatch": {
        "Comment": "",
        "Changes": [
            {
                "Action": "UPSERT",
                "ResourceRecordSet": {
                    "Name": "klautsan.net.",
                    "Type": "A",
                    "TTL": 60,
                    "ResourceRecords": [
                        {
                            "Value": ${public_ip}
                        }
                    ]
                }
            }
        ]
    }
}
EOF
)

echo ${request}

echo "Binding DNS record for instance"
aws route53 change-resource-record-sets --cli-input-json "${request}"
