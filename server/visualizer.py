from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_ollama import ChatOllama
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
import re, os
import sys
# enable the agurment parsing from remote machine
if len(sys.argv) < 2:
    sys.exit(1)

WORKING_DIR = sys.argv[1] 
# Set up the LightRAG instance
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name='phi4:14b-q8_0',
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embed(
            texts, embed_model="nomic-embed-text", host="http://localhost:11434"
        ),
    ),

)
# Load the document into the LightRAG instance
def get_txt_file_from_dir(directory):
    """Finds the first `.txt` file in the given directory."""
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            return os.path.join(directory, file)
    return None 

document_path = get_txt_file_from_dir(WORKING_DIR)
if document_path:
    with open(document_path, "r", encoding="utf-8") as f:
        rag.insert(f.read())
# Normalize the nodes in the cluster
# This function will rename nodes in the cluster to avoid duplicates
def normalize_nodes_in_cluster(dot_code):
    def process_cluster(match):
        cluster_id = match.group(1)  # e.g., cluster_3
        body = match.group(2)

        cluster_num = cluster_id.split("_")[1]

        # Find all actor_*
        actor_defs = re.findall(r'(actor_\d+)\s*\[label="[^"]+"\]', body)
        entity_defs = re.findall(r'(entity_\d+_\d+)\s*\[label="[^"]+"\]', body)

        # Create renaming maps
        actor_map = {old: f"actor_{cluster_num}_{i+1}" for i, old in enumerate(sorted(set(actor_defs)))}
        entity_map = {old: f"entity_{cluster_num}_{i+1}" for i, old in enumerate(sorted(set(entity_defs)))}

        # Replace in body
        for old, new in actor_map.items():
            body = re.sub(rf'\b{old}\b', new, body)

        for old, new in entity_map.items():
            body = re.sub(rf'\b{old}\b', new, body)

        return f"subgraph {cluster_id} {{ {body} }}"

    # Update all clusters
    updated_code = re.sub(r'subgraph (cluster_\d+)\s*{\s*(.*?)\s*}', process_cluster, dot_code, flags=re.DOTALL)
    return updated_code
# Replace clusters with actors in the DOT code
def replace_clusters_with_actors(dot_code: str, cluster_actor_map: dict) -> str:
    new_edges = []

    # Match DOT edges with cluster and full label
    pattern = r'(cluster_\d+)\s*->\s*(cluster_\d+)\s*\[label\s*=\s*"([^"]+)"\]'

    for match in re.finditer(pattern, dot_code):
        src_cluster, dst_cluster, full_label = match.groups()

        # Get corresponding actors
        src_actor = cluster_actor_map.get(src_cluster, ['MISSING_SRC'])[0]
        dst_actor = cluster_actor_map.get(dst_cluster, ['MISSING_DST'])[0]

        # Format new DOT edge line with full label
        edge_line = (
            f'{src_actor} -> {dst_actor} [label="{full_label}", '
            f'ltail={src_cluster}, lhead={dst_cluster}];'
        )
        new_edges.append(edge_line)

    return "\n".join(new_edges)
# Step 1: Identify the steps in the document
def identity_steps():
    num_of_steps = rag.query("Scan the whole doucment, how many steps in the document, what are they? just give the number of steps and list name of steps with its number, no details, don't give any explantion or summary, pay attention to the headers each passage",param=QueryParam(mode="global"))

    
    few_shot_examples = [
        {
            "input": "Step 1: Prepare Ingredients\nStep 2: Cook Meat\nStep 3: Serve Dish",
            "output": """
                subgraph cluster_1 {{ label="Prepare Ingredients" }}
                subgraph cluster_2 {{ label="Cook Meat" }}
                subgraph cluster_3 {{ label="Serve Dish" }}
            """
        },
        {
            "input": "Preheat Oven\nMix Batter\nBake Cake",
            "output": """
                subgraph cluster_1 {{ label="Preheat Oven" }}
                subgraph cluster_2 {{ label="Mix Batter" }}
                subgraph cluster_3 {{ label="Bake Cake" }}
            """
        }
    ]

    example_template = """
    Input steps:
    {input}

    Output DOT code:
    {output}
    """

    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template=example_template
    )
    
    cot_instructions = """
    You are tasked with converting a list of procedural steps into DOT code for a flowchart. 
    Each step should become a subgraph with a descriptive label. Follow these steps:
    1. Read the list of steps provided.
    2. For each step, create a subgraph in DOT syntax (e.g., 'subgraph cluster_X {{ label="Step Name" }}')., keep the number in front of the step name
    3. Number the subgraphs sequentially (cluster_1, cluster_2, etc.).
    4. Ensure that you don't include any other details, just the subgraph definitions.
    4. Ensure the output is valid DOT code that can be rendered as a flowchart, output is dot code, no more details needed, no explanation or irrlevant part or description.

    Now, process the following steps and output the DOT code:
    {steps}
    """
    
    few_shot_prompt = FewShotPromptTemplate(
    examples=few_shot_examples,
    example_prompt=example_prompt,
    prefix="Here are some examples of converting steps to DOT code:\n",
    suffix=cot_instructions,
    input_variables=["steps"],
    example_separator="\n---\n"
    )
    
    # Use num_of_steps directly as the input
    final_prompt = few_shot_prompt.format(steps=num_of_steps)
    llm = ChatOllama(model = 'deepseek-r1:14b-qwen-distill-q8_0', temperature=0.0)
    dot_code_output = llm.invoke(final_prompt)
    dot_code_output_step1 = dot_code_output.content
    dot_code_output_step1 = re.sub(r"<think>\s*.*?\s*</think>", "", dot_code_output.content, flags=re.DOTALL).strip()
    dot_code_output_step1 = re.sub(r"dot", "", dot_code_output_step1)
   
    matches = re.findall(r'subgraph cluster_(\d+)\s*{ label="(.*?)"', dot_code_output_step1)
    num_of_steps = "\n".join([f"{num}. {label}" for num, label in matches])
    step_names = num_of_steps.split("\n")
    return step_names,num_of_steps, dot_code_output_step1
# Step 2: Identify the entities and relations in each step
def identify_entities(step_names):
    step_details = {}

    for step in step_names:
        query_step2 = (
            f"For the step '{step}' in the document, identify: "
            "1. The actors list all the actors (who or what performs the action, if specified, if not, just use the pronouce 'you' or imply the person, actor, whom, what do the actions), keep the same actor if the same actor is used in the same step"
            "2. The main action, procedure, or event, etc "
            "3. The entities (nouns/objects/things involved, excluding the actor), "
            "4. Relevant info (conditions, details, or constraints). "
            "Return the response in this format:\n"
            "Actor: <actor>\nAction: <action>\nEntities: <entity1>, <entity2>, ...\nRelevant Info: <info>"
        )
        details = rag.query(query_step2, param=QueryParam(mode="global"))
        step_details[step] = details
    example_template = """
    Input step and details:
    {input}

    Output DOT code:
    {output}
    """

    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template=example_template
    )

    # Chain-of-thought instructions with actor
    cot_instructions = """
    You are converting a procedural step and its details into DOT code for a flowchart. 
    Each step is a subgraph containing nodes for actor, action, entities, and relevant info. Follow these rules:
    1. Read the step name and its details (Actor, Action, Entities, Relevant Info).
    2. Create a subgraph with the step name as the label: 'subgraph cluster_X {{ label="Step Name"; }}'.
    3. Inside the subgraph, add nodes:
    - If an actor is specified (not 'None'): 'actor_X [label="Actor"];', if not defined, use  'You', if defined, specify the actor,  actor_11 [label="You"], actor_12 [label="Other guy"] etc.
    - For the action: 'action_X [label="Action"];' for example, action_1, action_2, etc.
    - For each entity: 'entity_Y [label="Entity"];'
    - For relevant info (if not empty): 'info_Z [label="Info"];'
    4. Use sequential cluster numbering (cluster_1, cluster_2, etc.) based on step order.
    5. Use unique node IDs within each subgraph (e.g., actor_1, action_1, entity_1, info_1 for cluster_1).
    6. Ensure double quotes around all label text and semicolons after each node.
    7. Indent nodes for readability, but keep the subgraph on multiple lines.
    8. Only output the dotcode, no other details, no explanation, no description

    Process this step and its details and output the DOT code, remove any other details like description or explanation:
    {step_details}
    """
    few_shot_examples = [
    {
        "input": (
            "Step 1: Brian and Stewie cook meat\n"
            "Actor1: Brian\n"
            "Actor2: Stewie\n"
            "Action: Cooks\n"
            "Entities: Meat\n"
            "Relevant Info: On medium heat"
        ),
        "output": """
            'subgraph cluster_1 {{ label="Brian and Stewie cook meat";\n'
            '  actor_1 [label="Brian"];\n'
            '  actor_2 [label="Stewie"];\n'
            '  action_1 [label="Cooks"];\n'
            '  entity_1 [label="Meat"];\n'
            '  info_1 [label="On medium heat"];\n'
            '}}'
        """
    },
    {
        "input": (
            "Step 2: Boil the water\n"
            "Actor: You\n"
            "Action: Boil\n"
            "Entities: Water\n"
            "Relevant Info: High heat"
        ),
        "output": """
            'subgraph cluster_2 {{ label="Boil the water";\n'
            '  actor_2 [label="You"];\n'
            '  action_2 [label="Boil"];\n'
            '  entity_2 [label="Water"];\n'
            '  info_2 [label="High heat"];\n'
            '}}'
        """
    }
]
    # Combine into few-shot prompt
    few_shot_prompt = FewShotPromptTemplate(
        examples=few_shot_examples,
        example_prompt=example_prompt,
        prefix="Here are examples of converting step details to DOT code with nodes:\n",
        suffix=cot_instructions,
        input_variables=["step_details"],
        example_separator="\n---\n"
    )
    dot_code_output_step2 = []
    for step in step_names:
        step_input = f"{step}\n{step_details[step]}"
        final_prompt = few_shot_prompt.format(step_details=step_input)
        llm = ChatOllama(model = 'phi4:14b-q8_0', temperature=0.0)
        dot_code_output = llm.invoke(final_prompt)
        #dot_code_output = dot_code_output.content
        dot_code_output= re.sub(r"<think>.*?</think>", "", dot_code_output.content, flags=re.DOTALL).strip()
        dot_code_output_step2.append(dot_code_output)
    
    return step_details, dot_code_output_step2
# Step 3: Identify the relations between the entities in each step
def identify_relations(step_details, step_names):
    step_relations = {}

    for step_name, step_detail in zip(step_names, step_details.values()):
        query_step3 = (
            f"For the step '{step_name}' with details:\n{step_detail}\n"
            "Identify the direct relationships between the actors (there could be many actors or just 1) and entities, where the action is the relationship label "
            "Return in this format:\n"
            "From: <actor> To: <entity> Label: <action>\n"
            "If no actor, use 'You' or use the actor name at step details as the default actor depends on the context, if defined. Include one line per entity."
        )
        relations = rag.query(query_step3, param=QueryParam(mode="global"))
        step_relations[step_name] = relations
       # print(f"Step 3 - Relationships for '{step_name}':\n", relations)
        
    few_shot_examples = [
        {
            "input": (
                "Step 1: Brian and Stewie cook meat\n"
                "Actor: Brian, Stewie\nAction: Cooks\nEntities: Meat\nRelevant Info: On medium heat\n"
                "Relationships:\n"
                "From: Brian, Stewie To: Meat Label: Cooks"
            ),
            "output": """
                'subgraph cluster_1 {{ label="Brian and Stewie cook meat";\n'
                '  actor_1 [label="Brian"];\n'
                '  actor_2 [label="Stewie"];\n'
                '  entity_1 [label="Meat"];\n'
                '  actor_1 -> entity_1 [label="Cooks"];\n'
                '  actor_2 -> entity_1 [label="Cooks"];\n'
                '}}'
            """
        },
        {
            "input": (
                "Step 2: Boil the water\n"
                "Actor: None\nAction: Boil\nEntities: Water\nRelevant Info: High heat\n"
                "Relationships:\n"
                "From: Process To: Water Label: Boil"
            ),
            "output":"""
                'subgraph cluster_2 {{ label="Boil the water";\n'
                '  actor_2_1 [label="You"];\n'
                '  entity_2 [label="Water"];\n'
                '  actor_2_1 -> entity_2 [label="Boil"];\n'
                '}}'
            """
        },
        {
            "input": (
                "3: Add salt and pepper\n"
                "Actor: None\nAction: Add\nEntities: Salt, Pepper\nRelevant Info: To taste\n"
                "Relationships:\n"
                "From: Process To: Salt Label: Add\n"
                "From: Process To: Pepper Label: Add"
            ),
            "output": """
                'subgraph cluster_3 {{ label="Add salt and pepper";\n'
                '  actor_3 [label="you"];\n'
                '  entity_3_1 [label="Salt"];\n'
                '  entity_3_2 [label="Pepper"];\n'
                '  actor_3 -> entity_3_1 [label="Add"];\n'
                '  actor_3 -> entity_3_2 [label="Add"];\n'
                '}}'
            """
        }
    ]

    # Example template
    example_template = """
    Input step, details, and relationships:
    {input}

    Output DOT code with nodes and edges:
    {output}
    """

    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template=example_template
    )

    # Chain-of-thought instructions
    cot_instructions = """
    You are converting a procedural step, its details, and relationships into DOT code for a flowchart. 
    Each step is a subgraph with nodes and directed edges. Follow these rules:
    1. Read the step name, details (Actor, Action, Entities, Relevant Info), and relationships.
    2. Create a subgraph: 'subgraph cluster_X {{ label="Step Name"; }}'.
    3. Add nodes:
    - For Actor (use 'You' if 'None'): 'actor_X [label="Actor"];' 
    - For each Entity: 'entity_X_Y [label="Entity"];'
    4. Add edges from relationships in the format 'From: <actor> To: <entity> Label: <action>'  :
    - '<actor_X> -> <entity_X_Y> [label="Action"];'
    5. Use sequential numbering for clusters (cluster_1, cluster_2, etc.).
    6. Use unique node IDs within each subgraph (e.g., actor_1, entity_1_1, info_1).
    7. Ensure double quotes around labels and semicolons after each statement.
    8. Omit Action node as its already represented as relation 
    9. Only output the dotcode(remove the info in final output), no other details

    Process this step, its details, and relationships, and output the DOT code:
    {step_data}
    """
    few_shot_prompt = FewShotPromptTemplate(
        examples=few_shot_examples,
        example_prompt=example_prompt,
        prefix="Here are examples of converting step details and relationships to DOT code:\n",
        suffix=cot_instructions,
        input_variables=["step_data"],
        example_separator="\n---\n"
    )
    
    dot_code_output_step3 = []
    
    for step_name in step_names:
        relations = step_relations.get(step_name)
        step_input = f"{step_name}\nRelationships:\n{relations}"
        final_prompt = few_shot_prompt.format(step_data=step_input)
        llm = ChatOllama(model = 'phi4:14b-q8_0', temperature=0.0)
        dot_code_output = llm.invoke(final_prompt)
        dot_code_output= re.sub(r"<think>.*?</think>", "", dot_code_output.content, flags=re.DOTALL).strip()
        dot_code_output = re.search(r'(subgraph\s+cluster_\d+\s*\{.*?\})', dot_code_output, re.DOTALL).group(1)
        dot_code_output_step3.append(dot_code_output)


    dot_code_output_step3 = [dot_code_output.replace("dot", "") for dot_code_output in dot_code_output_step3]
    dot_code_output_step3 = [re.sub(r'```', '', subgraph).strip() for subgraph in dot_code_output_step3]
    dot_code_output_step3 = [re.sub(r'\n', '', subgraph).strip() for subgraph in dot_code_output_step3]
    actor_dict = {}
    merged_dot_code = []
    for dot_code in dot_code_output_step3:     
        # Normalize the nodes in the cluster
        dot_code = normalize_nodes_in_cluster(dot_code)
       # print("Normalized DOT code:\n", dot_code)
        merged_dot_code.append(dot_code)
        # Extract cluster ID like `cluster_1`
        cluster_id_match = re.search(r'subgraph\s+(cluster_\d+)', dot_code)
        cluster_id = cluster_id_match.group(1) if cluster_id_match else "unknown_cluster"

        # Extract actor node names like actor_1, actor_2 (before the label)
        actor_nodes = re.findall(r'(actor_\d+_\d+)\s*\[label="[^"]+"\]', dot_code)

        unique_actors = list(set(actor_nodes))
        actor_dict[cluster_id] = unique_actors
   # print("Step 3 - Actor dictionary:\n", actor_dict)
    dot_code_output_step3 = "\n".join(merged_dot_code)
        
    return dot_code_output_step3, actor_dict
# Step 4: Identify the relations between the steps
def identify_steps_relations(num_of_steps, dot_code_output_step1):
    query_step4 = (
    f"Document contains these steps:\n{num_of_steps}\n"
    "Identify relationships between steps EXACTLY in this format:\n"
    "From: <X> To: <Y> Label: \"<DESCRIPTIVE_RELATIONSHIP>\"\n"
    "Where:\n"
    "- X and Y are ONLY existing step numbers from the list above\n"
    "- Labels MUST include specific conditions or actions described in the document\n"
    "- Keep labels concise and descriptive (around 7 words max)\n"
    "- Use these patterns when appropriate:\n"
    "  * \"if [condition]\" for conditional flows\n"
    "  * \"requires [dependency]\" for dependencies\n"
    "  * \"followed by [next action]\" for sequences\n"
    "- You do NOT need to include labels like \"branches when [...]\" or \"merges after [...]\"\n"
    "- Instead, simply list all relevant pairwise relationships — multiple outputs from a step imply a decision; multiple inputs to a step imply a merge\n"
    "- A step may appear multiple times (as source or destination)\n\n"
    "Examples:\n"
    "From: 1 To: 4 Label: \"if stateful behavior required\"\n"
    "From: 3 To: 9 Label: \"requires service discovery completion\"\n"
    "From: 4 To: 5 Label: \"when external exposure needed\"\n"
    "From: 6 To: 7 Label: \"followed by TLS security setup\"\n"
    "From: 7 To: 8 Label: \"followed by observability configuration\"\n"
    "From: 2 To: 4 Label: \"if user is authenticated\"\n"
    "From: 2 To: 5 Label: \"if user is anonymous\"\n"
    "From: 8 To: 9 Label: \"after observability complete\"\n"
    "From: 7 To: 9 Label: \"after security setup\"\n"
)
    inter_step_relations = rag.query(query_step4, param=QueryParam(mode="mix"))
  #  print("Step 4 - Raw inter-step relationships:\n", inter_step_relations)

    few_shot_examples = [
    {
        "input": (
            "Steps:\n"
            "1. Check water level\n"
            "2. Boil the water\n"
            "Relationships:\n"
            'From: 1 To: 2 Label: "after verifying minimum level"'
        ),
        "output": (
            'cluster_1 -> cluster_2 [label="after verifying minimum level"];'
        )
    },
    {
        "input": (
            "Steps:\n"
            "1. Boil the water\n"
            "2. Check temperature\n"
            "3. Add ingredients\n"
            "4. Combine components\n"
            "Relationships:\n"
            'From: 1 To: 2 Label: "until reaching 100°C"\n'
            'From: 2 To: 3 Label: "if temperature maintained"\n'
            'From: 2,3 To: 4 Label: "merge cooked elements"'
        ),
        "output": (
            'cluster_1 -> cluster_2 [label="until reaching 100°C"];\n'
            'cluster_2 -> cluster_3 [label="if temperature maintained"];\n'
            'cluster_2, cluster_3 -> cluster_4 [label="merge cooked elements"];'
        )
    },
    {
        "input": (
            "Steps:\n"
            "1. Prepare ingredients\n"
            "2. Marinate meat\n"
            "3. Boil vegetables\n"
            "Relationships:\n"
            'From: 1 To: 2 Label: "requires pre-cut components"\n'
            'From: 1 To: 3 Label: "parallel cooking process"'
        ),
        "output": (
            'cluster_1 -> cluster_2 [label="requires pre-cut components"];\n'
            'cluster_1 -> cluster_3 [label="parallel cooking process"];'
        )
    },
    {
        "input": (
            "Steps:\n"
            "1. Sear protein\n"
            "2. Simmer sauce\n"
            "3. Plate dish\n"
            "Relationships:\n"
            'From: 1 To: 3 Label: "when caramelized crust forms"\n'
            'From: 2 To: 3 Label: "after reducing by 50%"'
        ),
        "output": (
            'cluster_1 -> cluster_3 [label="when caramelized crust forms"];\n'
            'cluster_2 -> cluster_3 [label="after reducing by 50%"];'
        )
    },
    {
        "input": (
            "Steps:\n"
            "1. Initialize system\n"
            "2. Check dependencies\n"
            "3. Start services\n"
            "Relationships:\n"
            'From: 1 To: 2 Label: "with configuration loaded"\n'
            'From: 2 To: 3 Label: "if all checks pass"\n'
            'From: 2 To: 4 Label: "when missing components"'
        ),
        "output": (
            'cluster_1 -> cluster_2 [label="with configuration loaded"];\n'
            'cluster_2 -> cluster_3 [label="if all checks pass"];\n'
            'cluster_2 -> cluster_4 [label="when missing components"];'
        )
    }
]
    example_template = """
    Input steps and relationships:
    {input}

    Output DOT edges between subgraphs:
    {output}
    """

    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template=example_template
    )

    # Chain-of-thought instructions
    cot_instructions = """
    You are generating DOT code edges to connect subgraphs representing procedural steps, using the Step 1 DOT code for context. 
    Follow these rules:
    1. Read the list of steps, Step 1 DOT code (subgraph titles), and relationships in the format 'From: <step_number> To: <step_number> Label: <relationship>'.
    2. For each relationship, create an edge between subgraphs:
    - 'cluster_X -> cluster_Y [label="Enhanced Relationship"];'
    3. Use the Step 1 DOT code to ensure cluster labels match the steps (e.g., cluster_1 for step 1).
    4. Enhance the relationship label based on context and type, using the subgraph titles for specificity(relationship lables could be used  with synonyms, act accordingly):
    - 'followed by' -> e.g., 'followed by to <cluster_Y label>'
    - 'if' -> e.g., 'if <cluster_X condition>'
    - 'depends on' -> e.g., 'depends on <cluster_X completion>'
    - 'parallel' -> e.g., 'parallel with <cluster_Y label>'
    - 'synchronize' -> e.g., 'synchronize with <cluster_X label>'
    - 'except' -> e.g., 'except if <cluster_X condition>'
    - 'trigger' -> e.g., 'triggers <cluster_Y label>'
    4. Format labels as: "<Relation Type>: <Brief Reason>"
    5. Output one edge per line.
    6. Keep the dot syntax (cluster_X -> cluster_Y [label="Relationship"];) and ensure the labels are in double quotes.
    6.Generate DOT edges following ALL rules, no more details needed, no explanation or irrlevant part or description (this step is crucial, it is expected output).
    Process these steps, Step 1 DOT code, and relationships, and output the DOT edges:
    {relations}
    """

    few_shot_prompt = FewShotPromptTemplate(
    examples=few_shot_examples,
    example_prompt=example_prompt,
    prefix="Here are examples of converting inter-step relationships to DOT edges with Step 1 DOT code:\n",
    suffix=cot_instructions,
    input_variables=["relations"],
    example_separator="\n---\n"
)
    step4_input = f"Steps:\n{num_of_steps}\nStep 1 DOT code:\n{dot_code_output_step1}\nRelationships:\n{inter_step_relations}"
    final_prompt = few_shot_prompt.format(relations=step4_input)
    llm = ChatOllama(model = 'deepseek-r1:14b-qwen-distill-q8_0', temperature=0.2)
    dot_code_output_step4 = llm.invoke(final_prompt)

    # extract dot code
    dot_code_output_step4 = re.sub(r"<think>\s*.*?\s*</think>", "", dot_code_output_step4.content, flags=re.DOTALL).strip()
    dot_code_output_step4 = validate_dot_code(dot_code_output_step4)
    return dot_code_output_step4
# Step 5: Validate and beautify the DOT code from step 4
def validate_dot_code(dot_code_output_step4):
    message_template = '''
    You are a DOT code expert. Your task is to validate, correct, and beautify the DOT graph based on the following edge descriptions.

    Instructions:
    - Convert the list of edges into proper DOT syntax.
    - Ensure all node and edge labels follow DOT format.
    - Remove any unnecessary and wrong characters or syntax.
    - Do not explain or describe the output. Only return the complete, corrected DOT code.
    
    This is the expected output format with no other details (just digraph and edge list), remove the  defining cluster part or explanation, only relations among clusters, just the code, no more details needed.:
    digraph G {{
    cluster_1 -> cluster_10 [label="If stateful behavior is required"];
    cluster_2 -> cluster_3 [label="Followed by Kubernetes orchestration"];
    cluster_3 -> cluster_4 [label="Requires service discovery completion"];
    cluster_4 -> cluster_5 [label="When external exposure is needed"];
    cluster_5 -> cluster_6 [label="Followed by observability configuration"];
    cluster_6 -> cluster_7 [label="Follows metrics setup with observability tools"];
    cluster_7 -> cluster_8 [label="Followed by deployment automation verification"];
    cluster_8 -> cluster_9 [label="Merges after observability complete"];
    cluster_9 -> cluster_10 [label="If persistence is ensured"];
    cluster_10 -> cluster_11 [label="Requires disaster recovery planning"];
}}
    Edge list to process:
    {dot_code_output_step4}    
    '''
    message = message_template.format(dot_code_output_step4=dot_code_output_step4)
    llm = ChatOllama(model = 'phi4:14b-q8_0', temperature=0.2)
    dot_code_output_step4 = llm.invoke(message)
    dot_code_output_step4 = re.sub(r"<think>.*?</think>", "", dot_code_output_step4.content, flags=re.DOTALL).strip()
    return dot_code_output_step4

def combine_dot_code(dot_code_output_step3: str, relations: str) -> str:
    return f"""digraph G {{
    compound=true;
    rankdir=TB;

    {dot_code_output_step3}

    {relations}
}}"""
# Main function to run the steps
if __name__ == "__main__":
    # identify the steps in the document
    steps, num_of_steps,dot_code_output_step1  = identity_steps()
    # identify the entities and relations in each step
    step_details, dot_code_output_step2 = identify_entities(steps)
    # identify the relations between the entities in each step
    dot_code_output_step3, actor_dict=identify_relations(step_details,steps)
    # identify the relations between the steps
    dot_code_output_step4 = identify_steps_relations(num_of_steps, dot_code_output_step1)
    # merge the relations between the steps and the entities
    relations = replace_clusters_with_actors(dot_code_output_step4, actor_dict)
    # combine the dot code with the relations
    combined_dot_code = combine_dot_code(dot_code_output_step3, relations)
    print(combined_dot_code)
    # save the combined dot code to a file
    with open("combined_graph.txt", "a") as f: 
        f.write("\n======================\n")  
        f.write(combined_dot_code)     
        f.write("\n")                 