import paramiko

host = '10.200.16.96'
user = 'rdt'
password = 'SRV4l3x4ndr3'
public_key = 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCyHS5WUuUsavzxEuWN5oJIHrD6acdnvdPYU5x0fjqk+cC661GJGAPS9ualNz6CV3BFSy64vowpnLNPzIJOJhzM/ey7d9Wc4J4ioBAKS0HXjIyjTi3B1q9L+Wfpe0sW3EVnFE1W0b5IGTWtS97QsxIEWVaOPn0NCVkk8FKRTmTzpfK4ZG2p0pjoC5TcgXJg93GtJYrRgQByhmZhidJh3nOkQklbLTKfkucYoK404XrYvyzf4iYaA7Ulb/FVvTlTGWXEotew1+51gA0+tjA16ziX87qyYDjuo8pfcyobGt59i8ojyF7y7EmJMsSkn9nXiTp8OJhuaSSBUp0aDIBWWZgrwz5eqFFO2+XSYyFYPWPdjjPQlUpr0LzvgSdTbcAzBXevRKgA7ETUKCUNfyZt4wBnXRL6RFpShBXU/crbaCm7lk92A9Ykrl5QaBwrNGql1Nb+3I0siyDNxjHaMV4+TmXxTejzhpNkZmhIi53pdFTgx7vhB4zClaBt9BH3mvBpTcR6Eo65+Nd3T6c9uRSZE5k/pHHEr8Cz5nCGkE+6vPqSxd3EvCrp+eY5sObWCLZVVjexvYiOEEU40SAqH/U8R5N09zSySkNcqw8+DtHf3FEb2z3qfrtpoG07qlCVVEo1MeJeYGx+mee0Xbk6I3VTcY9H8cyVwZs72iPvENZMaYl4rw== alexcarvalho73@hotmail.com'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    print(f"Conectando a {host}...")
    ssh.connect(host, username=user, password=password)
    
    # Create .ssh dir and append key
    commands = [
        "mkdir -p ~/.ssh",
        "chmod 700 ~/.ssh",
        f'echo "{public_key}" >> ~/.ssh/authorized_keys',
        "chmod 600 ~/.ssh/authorized_keys"
    ]
    
    for cmd in commands:
        ssh.exec_command(cmd)
        
    print("Chave SSH adicionada com sucesso ao servidor!")
except Exception as e:
    print(f"Erro: {e}")
finally:
    ssh.close()
