import paramiko
import os

host = '10.200.16.96'
user = 'rdt'
password = 'SRV4l3x4ndr3'
remote_dir = '/home/rdt/CameraIA'

files_to_upload = ['detector.py', 'main.py']

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    print(f"Conectando a {host}...")
    ssh.connect(host, username=user, password=password)
    print("Conectado.")

    # 1. Upload dos arquivos corrigidos
    print("Enviando arquivos...")
    sftp = ssh.open_sftp()
    for file in files_to_upload:
        print(f"Enviando {file}...")
        sftp.put(file, f"{remote_dir}/{file}")
    sftp.close()
    print("Upload concluído.")

    # 2. Parar o serviço atual
    print("Parando o serviço...")
    stdin, stdout, stderr = ssh.exec_command(f'cd {remote_dir} && chmod +x parar.sh && ./parar.sh')
    print(stdout.read().decode())

    # 3. Iniciar o serviço novamente
    print("Iniciando o serviço...")
    stdin, stdout, stderr = ssh.exec_command(f'cd {remote_dir} && chmod +x iniciar.sh && ./iniciar.sh')
    print(stdout.read().decode())
    
    print("Verificando se o processo iniciou...")
    stdin, stdout, stderr = ssh.exec_command('ps aux | grep python3 | grep -v grep')
    print(stdout.read().decode())

except Exception as e:
    print(f"Erro: {e}")
finally:
    ssh.close()
