#!/bin/bash
while true
    do
        if [ -z $(curl -Is http://169.254.169.254/latest/meta-data/spot/termination-time | head -1 | grep 404 | cut -d \  -f 2) ]
            then
                logger "Running shutdown hook."
                pkill jupyter
                sync
                # Call your shutdown script here.
                break
            else
                # Spot instance not yet marked for termination.
                sleep 5
        fi
    done
