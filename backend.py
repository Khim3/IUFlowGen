import yaml, os, paramiko, re
from scp import SCPClient
import shutil
import json
import streamlit as st
import re
from langchain_ollama import ChatOllama
import pypdfium2 as pdfium
import requests
import streamlit as st
from langchain.schema import HumanMessage, SystemMessage, AIMessage

with open("./main.css",encoding="utf-8") as f:
    style_file_content = f.read()

def load_config(file_path = 'config.yaml'):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

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
    
ssh_client = create_ssh_client(config['server']['ip'], config['server']['user'], config['server']['password'])

def send_folder_to_remote(file_name):
    """Connects to a remote server via SSH and copies a local folder to a remote directory."""
    local_folder = f'./{file_name}'
    remote_folder = f'/home/tttung/Khiem/thesis/{file_name}'
    try:
        # Ensure local folder exists
        if not os.path.exists(local_folder):
            print(f"Error: Local folder '{local_folder}' does not exist.")
            return

        # Check if folder already exists on the remote server
        check_command = f'test -d "{remote_folder}" && echo "exists" || echo "not_exists"'
        stdin, stdout, stderr = ssh_client.exec_command(check_command)
        result = stdout.read().decode().strip()

        if result == "exists":
            print(f"Folder '{remote_folder}' already exists on remote machine. Skipping upload.")
            shutil.rmtree(local_folder)
            return remote_folder

        with SCPClient(ssh_client.get_transport()) as scp:
            scp.put(local_folder, remote_folder, recursive=True)
            print(f"Successfully copied folder '{local_folder}' to '{remote_folder}' on remote machine.")

        # Delete local folder after successful upload
        shutil.rmtree(local_folder)
        print(f"Local folder '{local_folder}' deleted successfully.")

    except Exception as e:
        print(f"Failed to copy folder: {e}")
    return remote_folder
    
def execute_query(query_text, working_dir):
    # Connect to the remote server and execute the query
    command = f'/home/tttung/Khiem/env/bin/python3 /home/tttung/Khiem/thesis/query.py  "{query_text}" "{working_dir}"'
    stdin, stdout, stderr = ssh_client.exec_command(command)
    # Fetch and print output
    output = stdout.read().decode()
    error = stderr.read().decode()
    print(output)
    return output
    
def render_dot_to_streamlit(dot_code):
    # Render the DOT code using D3.js and Graphviz in Streamlit
    escaped_dot_code = json.dumps(dot_code)

    html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Graphviz with D3.js (Styled)</title>
        <script src="//d3js.org/d3.v7.min.js"></script>
        <script src="https://unpkg.com/@hpcc-js/wasm@2.20.0/dist/graphviz.umd.js"></script>
        <script src="https://unpkg.com/d3-graphviz@5.6.0/build/d3-graphviz.js"></script>
        <style>
        {style_file_content}
        </style>
    </head>
    <body>

    <h2>📌 Graphviz Visualization</h2>
    <div id="graph-container">
        <div id="graph"></div>
    </div>

    <script>
        const scale = 1;

        function attributer(datum, index, nodes) {{
            var selection = d3.select(this);
            if (datum.tag === "svg") {{
                datum.attributes = {{
                    ...datum.attributes,
                    width: '1000',
                    height: '700',
                }};

                const px2pt = 3 / 4;
                const graphWidth = datum.attributes.viewBox.split(' ')[2] / px2pt;
                const graphHeight = datum.attributes.viewBox.split(' ')[3] / px2pt;

                const w = graphWidth / scale;
                const h = graphHeight / scale;
                const x = -(w - graphWidth) / 2;
                const y = -(h - graphHeight) / 2;

                const viewBox = `${{x * px2pt}} ${{y * px2pt}} ${{w * px2pt}} ${{h * px2pt}}`;
                selection.attr('viewBox', viewBox);
                datum.attributes.viewBox = viewBox;
            }}
        }}

        const dot = {escaped_dot_code};

        d3.select("#graph").graphviz()
            .fit(true)
            .attributer(attributer)
            .renderDot(dot);
    </script>

    </body>
    </html>
    """
    return st.components.v1.html(html_code, height=700, scrolling=True)

def clean_dot_code(dot_code: str) -> str:
    # Use regex to remove escaped quotes (e.g., \"Label\" → Label)
    cleaned_dot = re.sub(r'\\"(.*?)\\"', r'\1', dot_code)
    return cleaned_dot
    
def pdf_to_text(file_name,pdf_file_path,output_folder):
    # Converts a PDF file to text using pypdfium2
    try:
        pdf = pdfium.PdfDocument(pdf_file_path)
        text = ""
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            text += page.get_textpage().get_text_range() + "\n"
        text_file_path = os.path.join(output_folder, f"{file_name}.txt")
        with open(text_file_path, "w", encoding="utf-8") as text_file:
            text_file.write(text)
        return text_file_path
    
    except Exception as e:
        return f"An error occurred: {e}"
    
def create_folder(file_name):
    # Create a folder with the given file name
    folder_name = f"{file_name}"
    os.makedirs(f'./{folder_name}', exist_ok=True)
    print(f"Folder '{folder_name}' created successfully.")
    return folder_name
   
def process_text(WORKING_DIR):
    # Connect to the remote server and execute text processing
    command = f'/home/tttung/Khiem/env/bin/python3 /home/tttung/Khiem/thesis/text_processing.py "{WORKING_DIR}" '
    stdin, stdout, stderr = ssh_client.exec_command(command)
    # Fetch and print output
    output = stdout.read().decode()
    error = stderr.read().decode()
    return output

def run_chain (WORKING_DIR):
    # Connect to the remote server and execute pipeline
    command = f'/home/tttung/Khiem/env/bin/python3 /home/tttung/Khiem/thesis/visualizer.py  "{WORKING_DIR}"'
    stdin, stdout, stderr = ssh_client.exec_command(command)
    # Fetch and print output
    output = stdout.read().decode()
    error = stderr.read().decode()
    print(output)
    return output
    
def display_pdf(uploaded_file):
    # Display the uploaded PDF file using pypdfium2
    if uploaded_file is not None:
        try:
            # Read uploaded PDF
            pdf_bytes = uploaded_file.read()
            pdf = pdfium.PdfDocument(pdf_bytes)
            num_pages = len(pdf)

            # Reset page number when a new file is uploaded
            if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file:
                st.session_state.current_page = 0
                st.session_state.last_uploaded_file = uploaded_file 

            # Define navigation functions
            def next_page():
                st.session_state.current_page = min(st.session_state.current_page + 1, num_pages - 1)

            def prev_page():
                st.session_state.current_page = max(st.session_state.current_page - 1, 0)

            # Create a styled container for displaying PDF
            ctn_pdf_show = st.container(border=True)
            with ctn_pdf_show:
                st.subheader(f"{uploaded_file.name}", divider="blue")

                # Layout for buttons and page display
                colpdf_page = st.columns([0.5, 0.15, 0.75, 0.15, 0.5])

                # Previous Page Button
                with colpdf_page[1]:
                    if st.button(" ", icon=":material/keyboard_double_arrow_left:", key="btn_prev_pdf", use_container_width=True):
                        prev_page()

                # Next Page Button
                with colpdf_page[3]:
                    if st.button(" ", icon=":material/keyboard_double_arrow_right:", key="btn_next_pdf", use_container_width=True):
                        next_page()

                # Centered Page Number Display
                empty_caption = colpdf_page[2].empty()

                # Centered PDF Display
                small_col = st.columns([0.2, 0.5, 0.2])
                with small_col[1]:
                    if num_pages > 0:
                        page = pdf[st.session_state.current_page]
                        image = page.render(scale=2).to_pil()
                        st.image(image)

                        # Display page number in center
                        empty_caption.caption(
                            f"<div style='text-align: center;'>Page {st.session_state.current_page + 1} of {num_pages}</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.warning("The PDF has no pages.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload a PDF file.")

def query(output_folder): 
    # Initialize the chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="Should you have any difficulty understanding this document. Ask me anything!"),
        ]
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
            st.write(message.content)
    # User enters a query
    user_input = st.chat_input("Your query:")
    if user_input:
        # Display user's message in chat
        with st.chat_message("user"):
            st.write(user_input)

        # Add user message to session
        st.session_state.messages.append(HumanMessage(content=user_input))

        # Process query remotely
        server_response = execute_query(user_input,output_folder)

        # Display assistant response dynamically
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Simulating a streaming effect
            for char in server_response:
                full_response += char
                message_placeholder.write(full_response + "▌")  # Typing effect

            # Remove the cursor after completion
            message_placeholder.write(full_response)
            print(full_response)

        # Store assistant's response in chat history
        st.session_state.messages.append(AIMessage(content=server_response))
        
def extract_clusters_and_relations(dot_code: str) -> str:
    # Extract all subgraphs
    subgraphs = re.findall(r'(subgraph cluster_\d+ \{.*?\})', dot_code, re.DOTALL)

    cleaned_subgraphs = []
    for sg in subgraphs:
        # Extract cluster number
        cluster_match = re.search(r'subgraph\s+(cluster_\d+)', sg)
        cluster_id = cluster_match.group(1) if cluster_match else "cluster_unknown"

        # Extract label
        label_match = re.search(r'label\s*=\s*"([^"]+)"', sg)
        label = label_match.group(1) if label_match else "Unnamed"

        # Extract actor nodes
        actor_nodes = re.findall(r'(actor_\d+_\d+)\s*\[label="[^"]+"\];', sg)
        actor_lines = [re.search(rf'({actor_id}\s*\[label="[^"]+"\];)', sg).group(1) for actor_id in actor_nodes]

        # Compose cleaned subgraph
        subgraph_clean = f'  subgraph {cluster_id} {{ label="{label}";\n    ' + "\n    ".join(actor_lines) + '\n  }'
        cleaned_subgraphs.append(subgraph_clean)

    # Extract all inter-cluster relations
    relations = re.findall(r'(actor_\d+_\d+ -> actor_\d+_\d+ \[label="[^"]+", ltail=cluster_\d+, lhead=cluster_\d+\];)', dot_code)

    # Combine all into final DOT
    output = "digraph G {\n  compound=true;\n  rankdir=TB;\n\n"
    output += "\n\n".join(cleaned_subgraphs)
    output += "\n\n  " + "\n  ".join(relations)
    output += "\n}"
    return output

def convert_clusters_to_nodes(dot_code):
    # Extract all clusters with their number and label
    cluster_headers = re.findall(
        r'subgraph\s+cluster_(\d+)\s*\{[^{}]*?label\s*=\s*"([^"]+)"', dot_code
    )

    # Extract all actors and map them to their cluster
    actor_cluster_map = {}
    for num, label in cluster_headers:
        cluster_block = re.search(
            rf'subgraph\s+cluster_{num}\s*\{{(.*?)\}}', dot_code, re.DOTALL
        )
        if cluster_block:
            actors = re.findall(r'(actor_' + re.escape(num) + r'_\d+)', cluster_block.group(1))
            for actor in actors:
                actor_cluster_map[actor] = f'node_{num}'

    # Create node label map
    cluster_labels = {
        f'node_{num}': label for num, label in cluster_headers
    }

    # Extract inter-cluster actor-to-actor edges
    edges = re.findall(
        r'(actor_\d+_\d+)\s*->\s*(actor_\d+_\d+)\s*\[label="([^"]+)"', 
        dot_code
    )

    # Begin new dot code
    new_dot = ['digraph G {', '    rankdir=LR;']

    # Add new nodes with labels
    for node_id, label in cluster_labels.items():
        new_dot.append(f'    {node_id} [label="{label}"];')

    # Add edges (converted from actors to nodes)
    for src, dst, label in edges:
        src_node = actor_cluster_map.get(src)
        dst_node = actor_cluster_map.get(dst)
        if src_node and dst_node:
            new_dot.append(f'    {src_node} -> {dst_node} [label="{label}"];')

    new_dot.append('}')
    return '\n'.join(new_dot)

def clean_dot_code(dot_code: str) -> str:
    # Clean DOT code by removing escaped double quotes.
    # Use regex to remove escaped quotes (e.g., \"Label\" → Label)
    cleaned_dot = re.sub(r'\\"(.*?)\\"', r'\1', dot_code)
    return cleaned_dot

def beatify_dot_code(dot_code):
    # Add custom styles to the DOT code
    insert_styles = """
    bgcolor="#EDEDED"; 
    node [
        style=filled,
        fillcolor="#FFF3E0",
        color="#FFB74D",  
        fontname="Helvetica",
    ];

    edge [
        fontname="Helvetica",
        color="#1976D2",   
        penwidth=2,
        arrowsize=1.2,
    ];
    """
    style_block = """
        style=filled;
        fillcolor="#E3F2FD";
        color="#90CAF9"; 
        fontname="Helvetica-Bold";
        penwidth=2;
    """
    dot_code = re.sub(r'(rankdir\s*=\s*\w+;)', r'\1' + insert_styles, dot_code)
    dot_code = re.sub(
        r'(subgraph\s+cluster_\d+\s*{)',
        lambda m: f'{m.group(1)}{style_block}',
        dot_code
    )
    return dot_code