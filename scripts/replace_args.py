import subprocess

source_file = "scripts/testscript.py"
destination = "root@64.228.25.163:/root/testscript.py"

# Copy the file to the remote machine
scp_command = [
    "scp",
    "-o", "StrictHostKeyChecking=no",
    "-P", "52323",
    source_file,
    destination
]

subprocess.run(scp_command)
print("File copied successfully.")

# Execute the script on the remote machine
ssh_command = [
    "ssh",
    "-o", "StrictHostKeyChecking=no",
    "-p", "52323",
    "root@64.228.25.163",
    "python", "/root/testscript.py"
]

try:
    result = subprocess.run(ssh_command, capture_output=True, text=True, check=True)
    print("Standard Output:")
    print(result.stdout)
    print("Standard Error:")
    print(result.stderr)
except subprocess.CalledProcessError as e:
    print("Error executing remote script:")
    print(e.stderr)
