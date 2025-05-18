import streamlit as st
from PIL import Image
import tempfile
from backend import *

with open("./main.css",encoding="utf-8") as f:
    style_file_content = f.read()
st.set_page_config(
    page_title="Thesis",  # Custom title shown in browser tab
    page_icon="🎓",         # Custom favicon (emoji or URL)
    layout="wide"             # "wide" for full-screen layout
)

def main():
    sidebar = st.sidebar
    st.title('IUFlowGen App :mortar_board:')
    st.write('This is a system to turn documents into comprehensible flowcharts')
    st.sidebar.title(':gear: Menu Settings')
    uploaded_file = sidebar.file_uploader("Upload a PDF file", type=["pdf", 'docx'])
    col1, col2 = st.sidebar.columns(2)
    with col1:
        view_chart = col1.toggle("Show chart", value=True)
        rankdir_option=col1.selectbox("Choose flow direction", options=["LR", "TB"], index=0) 
       
    with col2:
        view_doc = col2.toggle("Show doc", value=True)
        chart_mode = col2.radio("Choose view mode", options=["overview", "detailed"], index=0)
        
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None
    # main page
    
    if uploaded_file:    
        file_name = os.path.splitext(uploaded_file.name)[0]
        if view_doc:
            display_pdf(uploaded_file)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            temp_pdf.write(uploaded_file.getbuffer())  
            temp_pdf_path = temp_pdf.name
        output_folder = create_folder(file_name)
        text_file_path = pdf_to_text(file_name,temp_pdf_path, output_folder)
        
        # generate button
        generate_button = st.sidebar.button('Process document')
        if generate_button:
            ## change for testing, remove when done
            output_folder = send_folder_to_remote(file_name)
            process_text(output_folder)
            raw_code = run_chain(output_folder)
            with open ("raw_code.txt", "w") as f:
               f.write(raw_code)
            with open ('raw_code.txt', 'r') as f:
                raw_code = f.read()
            st.session_state.output_folder = output_folder
            dot_code = clean_dot_code(raw_code)
            st.session_state.dot_code = dot_code
            st.sidebar.success("Processing completed!")
        if view_chart and "dot_code" in st.session_state:
            dot= convert_clusters_to_nodes(st.session_state.dot_code)
            full_code = st.session_state.dot_code
            if chart_mode == "detailed":
                full_code = beatify_dot_code(full_code)
                dot_code_modified = re.sub(r"rankdir=\s*\w+;", f"rankdir={rankdir_option};", full_code)
                render_dot_to_streamlit(dot_code_modified)
            else:
                dot = beatify_dot_code(dot)
                dot = re.sub(r"rankdir=\s*\w+;", f"rankdir={rankdir_option};", dot)
                render_dot_to_streamlit(dot)
            # Save the full dot code to a text file
            with open("full_graph.txt", "w") as f:
                f.write(full_code)
            # Save the short dot code to a text file
            with open("short_graph.txt", "w") as f:
                f.write(dot)  

        if "output_folder" in st.session_state:
            query(st.session_state.output_folder)
         
if __name__ == "__main__":
    main()

