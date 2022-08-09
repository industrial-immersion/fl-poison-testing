#!/bin/bash

set -euo pipefail

docker_image='nkakouros/flwr-run:model'

run_local() {
    if [[ ! -d data ]]; then
        echo Downloading image set
        wget -q https://datashare.ed.ac.uk/download/DS_10283_3192.zip &&
        unzip DS_10283_3192.zip && rm DS_10283_3192.zip &&
        mkdir -p data &&
        tar -C data -x -z -f CINIC-10.tar.gz && rm CINIC-10.tar.gz;
    fi

    echo "Starting server"
    python3 runner.py --server --config configs/config.local.yml &
    sleep 5  # Sleep for 3s to give the server enough time to start

    echo "Starting client 1"
    python3 runner.py --client --poisoned --config configs/config.local.yml &
    echo "Starting client 2"
    python3 runner.py --client --poisoned --config configs/config.local.yml &

    # This will allow you to use CTRL+C to stop all background processes
    trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
    # Wait for all background processes to complete
    wait
}

debug=false
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --debug)
	    debug=true
	    ;;
    esac
    shift
done


if [[ "$debug" == true ]]; then
    set -x
fi

run_local
