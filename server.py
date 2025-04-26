from backend import *
from local import *

config = load_config()
ssh_client = create_ssh_client(config['server']['ip'], config['server']['user'], config['server']['password'])

def create_ssh_client(ip, username, password):
    """Creates and returns an SSH client connected to the specified server."""
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(ip, username=username, password=password)
        return client
    except Exception as e:
        print(f"Failed to connect: {e}")
        return None
    
def execute_query(query_text, working_dir):
    command = f'/home/tttung/Khiem/env/bin/python3 /home/tttung/Khiem/thesis/query.py  "{query_text}" "{working_dir}"'
    stdin, stdout, stderr = ssh_client.exec_command(command)
    # Fetch and print output
    output = stdout.read().decode()
    error = stderr.read().decode()
    print(output)
    return output

def list_model_options():

    ollama_options = list_ollama_models()
    if ollama_options == []:
        #save_text_message(get_session_key(), "assistant", "No models available, please choose one from https://ollama.com/library and pull with /pull <model_name>")
        st.warning("No ollama models available, please choose one from https://ollama.com/library and pull with /pull <model_name>")
    return ollama_options   

def process_text(working_dir):
    command = f'/home/tttung/Khiem/env/bin/python3 /home/tttung/Khiem/thesis/text_processing.py "{working_dir}" '
    stdin, stdout, stderr = ssh_client.exec_command(command)
    # Fetch and print output
    output = stdout.read().decode()
    error = stderr.read().decode()
    #print(output)
    return output

def run_chain (WORKING_DIR):
   # WORKING_DIR = '/home/tttung/Khiem/thesis/lv1_cooking_procedure'
    command = f'/home/tttung/Khiem/env/bin/python3 /home/tttung/Khiem/thesis/visualizer.py  "{WORKING_DIR}"'
    stdin, stdout, stderr = ssh_client.exec_command(command)

    # Fetch and print output
    output = stdout.read().decode()
    error = stderr.read().decode()
    dot_code = extract_dot_code(output)
    print(dot_code)
    return dot_code

def list_all_remote_folder():
    remote_path="/home/tttung/Khiem/thesis/"
    try:
        command = f'ls -d {remote_path}*/ 2>/dev/null'  # List only directories
        stdin, stdout, stderr = ssh_client.exec_command(command)

        # Read output and process
        output = stdout.read().decode().strip()
        error = stderr.read().decode().strip()

        if error:
            print(f"Error listing remote directories: {error}")
            return []

        # Extract folder names from full paths
        folders = [folder.rstrip('/').split('/')[-1] for folder in output.split("\n") if folder and "__pycache__" not in folder]
        return folders

    except Exception as e:
        print(f"Error: {e}")
        return []