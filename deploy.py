import paramiko
import os

host = '10.200.16.96'
user = 'rdt'
password = 'SRV4l3x4ndr3'
remote_dir = '/home/rdt/CameraIA'

files_to_upload = ['main.py', 'detector.py', 'index.html', 'requirements.txt']

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    print(f"Connecting to {host}...")
    ssh.connect(host, username=user, password=password)
    print("Connected.")

    # Create directory
    stdin, stdout, stderr = ssh.exec_command(f'mkdir -p {remote_dir}')
    stdout.read()
    
    # Check dependencies
    print("Checking dependencies...")
    stdin, stdout, stderr = ssh.exec_command('python3 --version')
    print("Python version:", stdout.read().decode().strip())
    
    stdin, stdout, stderr = ssh.exec_command('pip3 --version')
    print("Pip version:", stdout.read().decode().strip())
    
    stdin, stdout, stderr = ssh.exec_command('dpkg -l | grep libgl1-mesa-glx')
    print("libgl1 (needed for OpenCV):", stdout.read().decode().strip() or "Not found")

    # Upload files
    print("Uploading files...")
    sftp = ssh.open_sftp()
    for file in files_to_upload:
        if os.path.exists(file):
            print(f"Uploading {file}...")
            sftp.put(file, f"{remote_dir}/{file}")
        else:
            print(f"File {file} not found locally.")
    sftp.close()
    
    print("Upload complete!")

except Exception as e:
    print(f"Error: {e}")
finally:
    ssh.close()
