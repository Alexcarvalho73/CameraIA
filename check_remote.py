import paramiko

host = '10.200.16.96'
user = 'rdt'
password = 'SRV4l3x4ndr3'
remote_dir = '/home/rdt/CameraIA'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    ssh.connect(host, username=user, password=password)
    
    # Check venv
    stdin, stdout, stderr = ssh.exec_command(f'ls -la {remote_dir}')
    print("Files in dir:", stdout.read().decode())
    
    # Try to recreate venv and print output
    stdin, stdout, stderr = ssh.exec_command(f'cd {remote_dir} && python3 -m venv venv')
    print("Venv create stdout:", stdout.read().decode())
    err = stderr.read().decode()
    print("Venv create stderr:", err)
    
    # Try pip install if venv exists
    if not err or "Error" not in err:
        print("Running pip install...")
        stdin, stdout, stderr = ssh.exec_command(f'cd {remote_dir} && ./venv/bin/pip install -r requirements.txt')
        print("Pip stdout:", stdout.read().decode())
        print("Pip stderr:", stderr.read().decode())
        
except Exception as e:
    print("Error:", e)
finally:
    ssh.close()
