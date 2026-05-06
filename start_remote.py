import paramiko

host = '10.200.16.96'
user = 'rdt'
password = 'SRV4l3x4ndr3'
remote_dir = '/home/rdt/CameraIA'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    ssh.connect(host, username=user, password=password)
    
    # Executar o iniciar.sh
    stdin, stdout, stderr = ssh.exec_command(f"cd {remote_dir} && ./iniciar.sh")
    
    # Ler a saida
    print("Saida do comando:")
    print(stdout.read().decode())
    
except Exception as e:
    print("Erro:", e)
finally:
    ssh.close()
