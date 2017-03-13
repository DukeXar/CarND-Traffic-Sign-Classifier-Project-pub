#!/bin/bash

set -e
set -x

hostname='klautsan.net.'
hostedzoneid=$(aws route53 list-hosted-zones --query "HostedZones[?Name==\`${hostname}\`].Id" --output text)
public_ip=$(ec2metadata --public-ipv4)

request=$(cat <<EOF
{
    "HostedZoneId": "${hostedzoneid}",
    "ChangeBatch": {
        "Comment": "",
        "Changes": [
            {
                "Action": "UPSERT",
                "ResourceRecordSet": {
                    "Name": "${hostname}",
                    "Type": "A",
                    "TTL": 60,
                    "ResourceRecords": [
                        {
                            "Value": "${public_ip}"
                        }
                    ]
                }
            }
        ]
    }
}
EOF
)

aws route53 change-resource-record-sets --cli-input-json "${request}"
