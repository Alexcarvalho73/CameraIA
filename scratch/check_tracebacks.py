import paramiko

host = '10.200.16.96'
user = 'rdt'
password = 'SRV4l3x4ndr3'
remote_dir = '/home/rdt/CameraIA'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    ssh.connect(host, username=user, password=password, timeout=10)
    
    print("--- Buscando Tracebacks no server.log ---")
    stdin, stdout, stderr = ssh.exec_command(f'grep -C 5 "Traceback" {remote_dir}/server.log | tail -n 50')
    print(stdout.read().decode())

    print("\n--- Buscando Erros genéricos ---")
    stdin, stdout, stderr = ssh.exec_command(f'grep -i "error\|exception\|fail" {remote_dir}/server.log | tail -n 20')
    print(stdout.read().decode())

except Exception as e:
    print("Erro:", e)
finally:
    ssh.close()
