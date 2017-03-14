#!/bin/bash

set -e
set -x

for d in *_state; do
	aws s3 sync ./$d s3://ak-carnd/$d
done
#aws s3 sync ./data s3://ak-carnd/traffic-sign-data  --recursive
#aws s3 cp lenet.data-00000-of-00001 s3://ak-carnd/traffic-sign-network/
#aws s3 cp lenet.index s3://ak-carnd/traffic-sign-network/
#aws s3 cp lenet.meta s3://ak-carnd/traffic-sign-network/
