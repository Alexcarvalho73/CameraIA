#!/bin/bash
cd /home/rdt/CameraIA
echo "Iniciando o Motor de Visao IA..."
export LD_LIBRARY_PATH=/home/rdt/CameraIA/instantclient_21_1:$LD_LIBRARY_PATH
nohup ./venv/bin/python -u main.py > server.log 2>&1 &
echo $! > server.pid
echo "Motor rodando em segundo plano (PID: $(cat server.pid))."
echo "Acesse o dashboard em: http://10.200.16.96:5050"
