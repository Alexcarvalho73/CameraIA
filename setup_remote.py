import paramiko

host = '10.200.16.96'
user = 'rdt'
password = 'SRV4l3x4ndr3'
remote_dir = '/home/rdt/CameraIA'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    print(f"Connecting to {host}...")
    ssh.connect(host, username=user, password=password)
    
    def run_sudo_cmd(cmd):
        print(f"Running: {cmd}")
        stdin, stdout, stderr = ssh.exec_command(f'echo {password} | sudo -S {cmd}')
        out = stdout.read().decode()
        err = stderr.read().decode()
        if out: print(out)
        if err: print(err)

    print("Updating apt...")
    run_sudo_cmd("apt-get update")
    
    print("Installing dependencies...")
    run_sudo_cmd("apt-get install -y python3-pip python3-venv libgl1 libglib2.0-0")
    
    print("Setting up virtual environment...")
    ssh.exec_command(f'cd {remote_dir} && python3 -m venv venv')
    
    print("Installing python packages...")
    stdin, stdout, stderr = ssh.exec_command(f'cd {remote_dir} && ./venv/bin/pip install -r requirements.txt')
    print(stdout.read().decode())
    print(stderr.read().decode())

    print("Setup complete.")

except Exception as e:
    print(f"Error: {e}")
finally:
    ssh.close()
