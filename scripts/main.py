import paramiko
import sys
from scp import SCPClient

input_value = sys.argv[1]
# input_value = 34
# Assuming you have captured the user inputs in variables or a data structure

# gpu_ip = input('GPU IP: ')
# username = input('USERNAME: ')
# password = input('PASSWORD: ')

gpu_ip =  'ws.mec.ac.in'
username = 'project2-2023'
password = 'project@2023'
# ...

# Create an SSH client
ssh_client = paramiko.SSHClient()

# Automatically add the remote GPU to the list of known hosts
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# Connect to the remote GPU
ssh_client.connect(gpu_ip, username=username, password=password)

# Create an SCP client from the SSH client
scp = SCPClient(ssh_client.get_transport())

# Copy the local file to the remote server
local_path = 'scripts/testscript.py'
remote_path = 'py/'
scp.put(local_path, remote_path)

# Close the SCP and SSH clients
scp.close()
# Execute commands on the remote GPU
commandlist = 'cd py && python testscript.py ' + str(input_value)
stdin, stdout, stderr = ssh_client.exec_command(commandlist)

# Read the output from the remote GPU
output = stdout.read().decode()
print(output)
print("SSH closed")
# Close the SSH connection
ssh_client.close()
