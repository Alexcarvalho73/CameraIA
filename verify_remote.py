import paramiko

host = '10.200.16.96'
user = 'rdt'
password = 'SRV4l3x4ndr3'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    ssh.connect(host, username=user, password=password, timeout=10)
    
    stdin, stdout, stderr = ssh.exec_command('ls -la /home/rdt/CameraIA/venv/bin')
    print("Bin dir:", stdout.read().decode())
    
    stdin, stdout, stderr = ssh.exec_command('/home/rdt/CameraIA/venv/bin/pip list')
    print("Pip list:", stdout.read().decode())

except Exception as e:
    print("Error:", e)
finally:
    ssh.close()
