import paramiko
import time

host = '10.200.16.96'
user = 'rdt'
password = 'SRV4l3x4ndr3'
remote_dir = '/home/rdt/CameraIA'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    print(f"Conectando a {host}...")
    ssh.connect(host, username=user, password=password, timeout=10)
    print("Conectado!")

    # 1. Verificar processos python3
    print("\n--- Processos Python ---")
    stdin, stdout, stderr = ssh.exec_command('ps aux | grep python3 | grep -v grep')
    print(stdout.read().decode())

    # 2. Verificar log do servidor
    print("\n--- Ultimas 50 linhas do server.log ---")
    stdin, stdout, stderr = ssh.exec_command(f'tail -n 50 {remote_dir}/server.log')
    print(stdout.read().decode())

    # 3. Verificar se o arquivo server.pid existe e coincide com um processo
    print("\n--- Arquivo server.pid ---")
    stdin, stdout, stderr = ssh.exec_command(f'cat {remote_dir}/server.pid')
    pid = stdout.read().decode().strip()
    if pid:
        print(f"PID no arquivo: {pid}")
        stdin, stdout, stderr = ssh.exec_command(f'ps -p {pid}')
        print(stdout.read().decode())
    else:
        print("Arquivo server.pid não encontrado ou vazio.")

except Exception as e:
    print("Erro ao acessar o servidor:", e)
finally:
    ssh.close()
