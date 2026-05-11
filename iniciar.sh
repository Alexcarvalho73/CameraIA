#!/bin/bash
cd /home/rdt/CameraIA
source venv/bin/activate
export LD_LIBRARY_PATH=/home/rdt/CameraIA/instantclient_21_1:$LD_LIBRARY_PATH
export TNS_ADMIN=/home/rdt/CameraIA/DriveOracle
nohup python3 -u main.py > server.log 2>&1 &
echo $! > server.pid
