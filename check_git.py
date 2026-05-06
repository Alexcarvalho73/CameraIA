import paramiko

host = '10.200.16.96'
user = 'rdt'
password = 'SRV4l3x4ndr3'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    ssh.connect(host, username=user, password=password)
    
    # Check if git is installed
    stdin, stdout, stderr = ssh.exec_command("git --version")
    git_version = stdout.read().decode().strip()
    if git_version:
        print(f"Git installed: {git_version}")
    else:
        print("Git is NOT installed.")
        
    # Check connection to github
    stdin, stdout, stderr = ssh.exec_command("ssh -T -o StrictHostKeyChecking=no git@github.com 2>&1")
    github_response = stdout.read().decode().strip()
    if "successfully authenticated" in github_response or "Permission denied" in github_response:
        print("Conexão com GitHub: OK (Servidor consegue acessar a internet/porta SSH do Github).")
        print(f"Detalhes: {github_response}")
    else:
        print("Conexão com GitHub: Falhou ou não respondeu como esperado.")
        print(f"Detalhes: {github_response}")
        
except Exception as e:
    print("Erro ao verificar git:", e)
finally:
    ssh.close()
