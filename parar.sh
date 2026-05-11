#!/bin/bash
echo "Finalizando processos do Motor IA..."
pkill -9 -f main.py
rm -f /home/rdt/CameraIA/server.pid
echo "Motor encerrado com sucesso."
