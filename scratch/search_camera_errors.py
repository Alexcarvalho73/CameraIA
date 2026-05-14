import paramiko

host = '10.200.16.96'
user = 'rdt'
password = 'SRV4l3x4ndr3'
remote_dir = '/home/rdt/CameraIA'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    ssh.connect(host, username=user, password=password, timeout=10)
    
    # Procurar por erros específicos da camera_01 no log
    print("--- Buscando erros de camera_01 no server.log ---")
    stdin, stdout, stderr = ssh.exec_command(f'grep "camera_01" {remote_dir}/server.log | tail -n 20')
    print(stdout.read().decode())

    # Verificar se há mensagens de reconexão
    print("\n--- Buscando mensagens de reconexão ---")
    stdin, stdout, stderr = ssh.exec_command(f'grep "Reconectando" {remote_dir}/server.log | tail -n 20')
    print(stdout.read().decode())

    # Verificar o uso de CPU/Memória mais detalhado
    print("\n--- Uso de recursos ---")
    stdin, stdout, stderr = ssh.exec_command('top -b -n 1 | head -n 20')
    print(stdout.read().decode())

except Exception as e:
    print("Erro:", e)
finally:
    ssh.close()
