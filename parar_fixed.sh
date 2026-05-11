#!/bin/bash
cd /home/rdt/CameraIA
if [ -f server.pid ]; then
    PID=$(cat server.pid)
    echo "Parando o Motor de Visao IA (PID: $PID)..."
    kill $PID 2>/dev/null
    sleep 1
    # Verifica se ainda esta rodando e forca o kill se necessario
    if ps -p $PID > /dev/null; then
        echo "Processo $PID nao parou, forcando..."
        kill -9 $PID 2>/dev/null
    fi
    rm server.pid
    echo "Motor parado."
else
    echo "Nenhum PID encontrado em server.pid. Tentando pkill..."
    pkill -f "python.*main.py"
    echo "Processos finalizados via pkill."
fi
