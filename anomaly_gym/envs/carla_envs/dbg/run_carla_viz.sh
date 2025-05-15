#!/bin/bash

docker run -it --network="host" mjxu96/carlaviz:0.9.15 \
  --simulator_host localhost \
  --simulator_port 2000