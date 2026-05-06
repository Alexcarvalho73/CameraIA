import paramiko

host = '10.200.16.96'
user = 'rdt'
password = 'SRV4l3x4ndr3'
remote_dir = '/home/rdt/CameraIA'

iniciar_sh_content = """#!/bin/bash
cd /home/rdt/CameraIA
echo "Iniciando o Motor de Visao IA..."
nohup ./venv/bin/python main.py > server.log 2>&1 &
echo $! > server.pid
echo "Motor rodando em segundo plano (PID: $(cat server.pid))."
echo "Acesse o dashboard em: http://10.200.16.96:5050"
"""

parar_sh_content = """#!/bin/bash
cd /home/rdt/CameraIA
if [ -f server.pid ]; then
    PID=$(cat server.pid)
    echo "Parando o Motor de Visao IA (PID: $PID)..."
    kill $PID
    rm server.pid
    echo "Motor parado."
else
    echo "Nenhum PID encontrado. Tentando fechar todos os processos do motor..."
    pkill -f "python main.py"
    echo "Processos finalizados."
fi
"""

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    ssh.connect(host, username=user, password=password)
    
    # Criar iniciar.sh
    sftp = ssh.open_sftp()
    with sftp.file(f"{remote_dir}/iniciar.sh", "w") as f:
        f.write(iniciar_sh_content)
    
    # Criar parar.sh
    with sftp.file(f"{remote_dir}/parar.sh", "w") as f:
        f.write(parar_sh_content)
    sftp.close()
    
    # Dar permissao de execucao
    ssh.exec_command(f"chmod +x {remote_dir}/iniciar.sh {remote_dir}/parar.sh")
    
    print("Scripts iniciar.sh e parar.sh criados com sucesso na VM!")
    
except Exception as e:
    print("Erro ao criar scripts:", e)
finally:
    ssh.close()
