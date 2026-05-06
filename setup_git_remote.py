import paramiko

host = '10.200.16.96'
user = 'rdt'
password = 'SRV4l3x4ndr3'
remote_dir = '/home/rdt/CameraIA'
repo_url = 'https://github.com/Alexcarvalho73/CameraIA.git'

atualizar_sh_content = """#!/bin/bash
cd /home/rdt/CameraIA
echo "Buscando atualizacoes no GitHub..."
git pull origin main

echo "Reiniciando o servico..."
./parar.sh
sleep 2
./iniciar.sh
echo "Atualizacao e reinicio concluidos!"
"""

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    ssh.connect(host, username=user, password=password)
    
    commands = [
        f"cd {remote_dir} && git init",
        f"cd {remote_dir} && git remote add origin {repo_url}",
        f"cd {remote_dir} && git fetch --all",
        f"cd {remote_dir} && git branch -m main",
        f"cd {remote_dir} && git reset --hard origin/main"
    ]
    
    print("Configurando git na VM...")
    for cmd in commands:
        stdin, stdout, stderr = ssh.exec_command(cmd)
        stdout.read() # wait to finish
        
    sftp = ssh.open_sftp()
    with sftp.file(f"{remote_dir}/atualizar.sh", "w") as f:
        f.write(atualizar_sh_content)
    sftp.close()
    
    ssh.exec_command(f"chmod +x {remote_dir}/atualizar.sh")
    print("Script de atualizacao criado na VM.")

except Exception as e:
    print("Erro:", e)
finally:
    ssh.close()
