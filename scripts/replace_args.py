import paramiko
import sys
input_value = sys.argv[1]

def ssh_and_execute(hostname, username, source_file, destination_file, commands):
    # SSH connection
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=hostname, port=50483, username=username)
    
    # SCP transfer
    scp_client = ssh_client.open_sftp()
    scp_client.put(source_file, destination_file)
    scp_client.close()
    
    # Execute commands
    # command_output = ""
    for command in commands:
        stdin, stdout, stderr = ssh_client.exec_command(command)
        # stdout.channel.set_combine_stderr(True)
        # command_output += stdout.read().decode()

    stdin, stdout, stderr = ssh_client.exec_command(f"python3 testscripx.py {input_value}")
    stdout.channel.set_combine_stderr(True)
    performance_string = stdout.read().decode()

    # Commenting this out will output the standard print statements
    # stdin, stdout, stderr = ssh_client.exec_command('cat result.txt')
    # performance_string = stdout.read().decode()
    
    # Close SSH connection
    ssh_client.close()
    
    return performance_string

# Example usage
gpu_hostname = '45.142.208.117'
gpu_username = 'root'
local_file_path = 'scripts/scaleduptransformer.py'
# local_file_path = 'scripts/testscript.py'
remote_file_path = '/root/testscripx.py'
command_list = ['ls -l', 'pip install einops']

output = ssh_and_execute(gpu_hostname, gpu_username, local_file_path, remote_file_path, command_list)
print(output)
