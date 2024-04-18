import streamlit as st
import pandas as pd
import sys, os, json, time
sys.path.insert(0, '../')
import requests

from moonshot.api import (
    api_set_environment_variables,
)

### 'Global' Variables
url = ""
moonshot_path = "../moonshot/data/"
st.set_page_config(layout="wide")


env = {
    "CONNECTORS_ENDPOINTS": os.path.join(moonshot_path, "connectors-endpoints"),
    "CONNECTORS": os.path.join(moonshot_path, "connectors"),
    "RECIPES": os.path.join(moonshot_path, "recipes"),
    "COOKBOOKS": os.path.join(moonshot_path, "cookbooks"),
    "DATASETS": os.path.join(moonshot_path, "datasets"),
    "PROMPT_TEMPLATES": os.path.join(moonshot_path, "prompt-templates"),
    "METRICS": os.path.join(moonshot_path, "metrics"),
    "METRICS_CONFIG": os.path.join(moonshot_path, "metrics/metrics_config.json"),
    "CONTEXT_STRATEGY": os.path.join(moonshot_path, "context-strategy"),
    "RESULTS": os.path.join(moonshot_path, "results"),
    "DATABASES": os.path.join(moonshot_path, "databases"),
    "SESSIONS": os.path.join(moonshot_path, "sessions"),
}

api_set_environment_variables(env)

######## UTIL FUNCTIONS #########
def list_endpoints():
    """Get a list of endpoint and returns the ID
    as a list
    """
    results = []
    try:
        response = requests.get(f"{url}/v1/llm_endpoints/")
        get_endpoints = response.json()
        for endpoint in get_endpoints:
            results.append(endpoint["id"])
            
        return results
    except Exception:
        st.warning("Moonshot WebAPI is not connected.", icon="🔥")
        return results

def list_results(cookbook_ids):
    """Get a list of benchmark results and return as a dict
    """
    results = {}
    try:
        response = requests.get(f"{url}/v1/results")
        get_results = response.json()
        for result in get_results:
            id = result["metadata"]["id"]
            if cookbook_ids == None: 
                endpoints = result["metadata"]["endpoints"]
                results[id] = endpoints
            elif id in cookbook_ids:
                endpoints = result["metadata"]["endpoints"]
                results[id] = endpoints
        return results
    except Exception:
        st.warning("Moonshot WebAPI is not connected.", icon="🔥")
        return results

def list_connector_ids():
    """Get a list of connceots and returns the ID as a list
    """
    results = []
    try:
        response = requests.get(f"{url}/v1/connectors/")
        results = response.json()
        return results
    except Exception:
        st.warning("Moonshot WebAPI is not connected.", icon="🔥")
        return results

def list_sessions():
    """Get a list of active sessions and returns the ID as a list
    """
    results = []
    try:
        response = requests.get(f"{url}/v1/sessions/")
        get_sessions = response.json()
        for session in get_sessions:
            results.append(session["session_id"])
        return results
    except Exception:
        st.warning("Moonshot WebAPI is not connected.", icon="🔥")
        return results
    
def list_cookbook():
    """Get a list of cookbooks and returns the ID as a list
    """
    results = []
    try:
        response = requests.get(f"{url}/v1/cookbooks/")
        get_cookbooks = response.json()
        for cookbook in get_cookbooks:
            results.append(cookbook["id"])
        return results
    except Exception:
        st.warning("Moonshot WebAPI is not connected.", icon="🔥")
        return results

def get_cookbook():
    """Get a list of cookbook and its info"""
    results = []
    try:
        response = requests.get(f"{url}/v1/cookbooks/")
        results = response.json()
        return results
    except Exception:
        st.warning("Moonshot WebAPI is not connected.", icon="🔥")
        return results

def list_redteaming_recipes():
    """Get a list of recipes for red teaming and returns the ID as a list"""
    results = []
    try:
        response = requests.get(f"{url}/v1/recipes/redteaming/name")
        redteaming_recipes = response.json()
        return redteaming_recipes
    except Exception:
        st.warning("Moonshot WebAPI is not connected.", icon="🔥")
        return results

def get_endpoint_info(endpoint_id):
    """Get information about an endpoint"""
    try:
        response = requests.get(f"{url}/v1/llm_endpoints/{endpoint_id}")
        results = response.json()
        return results
    except Exception:
        st.warning("Moonshot WebAPI is not connected.", icon="🔥")
        return results

def get_session_info(session_id, get_history=False):
    try:
        url = st.session_state["moonshot_fastapi"]
        response = requests.get(f"{url}/v1/sessions/{session_id}?include_history={get_history}")
        return response.json()
    except Exception as e:
        st.warning(f"Unable to get session info {e}", icon="🔥")
        return False

def get_cookbook_by_id(cookbook_id):
    try:
        url = st.session_state["moonshot_fastapi"]
        response = requests.get(f"{url}/v1/cookbooks/{cookbook_id}")
        return response.json()
    except Exception as e:
        st.warning(f"Unable to get cookbook {e}", icon="🔥")
        return False
    
def get_exec_info_by_id(exec_id):
    try:
        url = st.session_state["moonshot_fastapi"]
        response = requests.get(f"{url}/v1/results/{exec_id}")
        return response
    except Exception as e:
        st.warning(f"Unable to get execution id {e}")
        return False
    
def get_benchmark_results(cookbook_ids=None, limit=3):
    """Limiting the results to 3 cookbook ID
    """
    results = list_results(cookbook_ids)
    final_results = {}
    
    selected_session = st.session_state["selected_session"]
    if selected_session == None or len(selected_session) == 0:
        st.warning("No session is activated.", icon="🔥")
        return
    
    session = get_session_info(selected_session)
    selected_endpoints = session["session"]["endpoints"]

    

    total_match = 0

    # key = flatten(cookbook_name, recipe_id, dataset_id, prompt_template_id)
    # value = [model, real results]

    ### The ridiculous number of for loops required to get to metrics
    for id in results:
        if results[id] == selected_endpoints: # must match fully 
            exec_id = get_exec_info_by_id(id).json()
            records = exec_id["results"]

            cookbook_id = id

            # Skipping all the recipes for this demo
            if "cookbooks" in records:
                for cookbook in records["cookbooks"]:
                    cookbook_name = cookbook["id"]
                    for recipe in cookbook["recipes"]:
                        recipe_id = recipe["id"]

                        for model in recipe["models"]:
                            model_id = model["id"]

                            #hack
                            if "toxic" in model_id:
                                continue
                        
                            for dataset in model["datasets"]:
                                dataset_id = dataset["id"]
                                for prompt_template in dataset["prompt_templates"]:
                                    real_results = prompt_template["metrics"]
                                    prompt_template_id = prompt_template["id"]
                                    if cookbook_id in final_results:
                                        key = f"{cookbook_id}, {cookbook_name}, {recipe_id}, {dataset_id}, {prompt_template_id}"

                                        if key in final_results[cookbook_id].keys():
                                            final_results[cookbook_id][key].append((model_id, real_results))
                                        else:
                                            final_results[cookbook_id][key] = [(model_id, real_results)]
                                        
                                    else:
                                        key = f"{cookbook_id}, {cookbook_name}, {recipe_id}, {dataset_id}, {prompt_template_id}"

                                        # Create the cookbook
                                        final_results[cookbook_id] = {}
                                        final_results[cookbook_id][key] = [(model_id, real_results)]

                total_match += 1

                if total_match >= limit:
                    return [final_results, selected_endpoints]

    return [final_results, selected_endpoints]

def get_chat_history():
    """Get history for all the chats"""
    selected_session = st.session_state["selected_session"]
    try:
        session_info = get_session_info(selected_session, get_history=True)
        chat_history = session_info["session"]["chat_history"]

        results = {}
        for chat in chat_history:
            if chat not in results.keys():
                results[chat] = []
            
            for chat_record in chat_history[chat]:
                prompt = chat_record["prompt"]
                predicted_result = chat_record["predicted_result"]

                results[chat].append([prompt, predicted_result])

        return results
    except Exception as e:
        st.warning(f"Unable to get execution id {e}")
        return False

def print_benchmark_results(cookbook_id, results_endpoint):
    exec_info = get_exec_info_by_id(cookbook_id).json()
    result_name = os.path.join(env["RESULTS"], exec_info["metadata"]["id"] + ".json")
    
    with open(result_name, "r") as f:
        data = json.load(f)

    cookbooks = {}
    start_time = data["metadata"]["start_time"]
    end_time = data["metadata"]["end_time"]
    duration = data["metadata"]["duration"]
    num_of_prompts = data["metadata"]["num_of_prompts"]

    metadata_table = "| Start Time | End Time | Duration | Number of Prompts Per Recipe | "
    metadata_table += "\n| :------------ | :------------ | :------------  | :------------ |"
    metadata_table += f"\n| {start_time} | {end_time} | {duration}  | {num_of_prompts} |"
    st.markdown(metadata_table)
    st.divider()
    
    table_column = ""

    # get the selected endpoints
    results = results_endpoint[0]
    selected_endpoints = results_endpoint[1]

    if "toxic" in selected_endpoints[-1]:
        selected_endpoints = selected_endpoints[:-1]

    for result in results:
        cookbook = result.split(",")[1]

        if cookbook not in cookbooks.keys():
            cookbooks[cookbook] = [(result, results[result])]
        else:
            cookbooks[cookbook].append((result, results[result]))

    for cookbook in cookbooks:  
        # create a new table
        st.markdown(f"### {cookbook}")
        endpoints_str = " | ".join(x for x in selected_endpoints)
        table_column = f"|  &nbsp; | {endpoints_str}"

        total_number_of_divider = len(selected_endpoints) + 1 
        table_column += "\n|"
        for i in range(total_number_of_divider):
            table_column += " :------------| "    
        
        table_column += "\n| **Recipe**  | &nbsp; | &nbsp; | "
        
        for each_combination in cookbooks[cookbook]:
            name = ", ".join(each_combination[0].split(",")[2:3])
            metrics = each_combination[1]

            metric_str = "|"
            for metric in metrics:
                real_metric = metric[1][0] ### hard-codedfor now, but we should discus show to proceed

                for metric_name in real_metric:
                    metric_str += f" {metric_name}: {real_metric[metric_name]} | "
                    break # skipping other metrics, only use the first one
            
            table_column += "\n| {0} {1}".format(name, metric_str)

        st.markdown(table_column, unsafe_allow_html=True)
        st.divider()
        
    st.divider()

    # Expand with the details of the results
    with st.expander("See details of the results"):
        st.write(data)

def print_cookbook_info():
    cookbooks = get_cookbook()
    st.write(cookbooks)

def send_prompt():
    selected_session = st.session_state["selected_session"]
    if selected_session == None or len(selected_session) == 0:
        st.warning("No session is activated.", icon="🔥")
        return
    
    # get the number of chat ID in this session
    session = get_session_info(selected_session)
    prompt = st.session_state["main_prompt"]

    if session == None:
        st.error("Create or activate a session using the sidebar options.", icon="❌")
        return False
    else:
        try:
            session_id = session["session"]["session_id"]
            payload = {
                "prompt": prompt,
                "history_length": 10
            }

            url = st.session_state["moonshot_fastapi"]
            response = requests.post(f"{url}/v1/sessions/{session_id}/prompt",
                                    json=payload)
            
            if response.status_code == 500:
                st.error("Unable to send to your model. Please check your endpoint settings.", icon="❌")
            
            start_red_teaming()
            return response 
        except Exception as e:
            st.warning(f"Unable to send prompt: {e}")
            return False

############ On Change Function ############
def run_automated_attack():
    selected_session = st.session_state["selected_session"]
    selected_attack_module = st.session_state["selected_attack_modules"]

    if selected_session == None or len(selected_session) == 0:
        st.warning("No session is activated.", icon="🔥")
        return

    if len(selected_attack_module) == 0:
        st.warning("Unable to run. No cookbook is selected.", icon="🔥")
        return
    
    # Get the Chat IDs and Create the tabs accordingly
    session = get_session_info(selected_session)
    selected_endpoints = session["session"]["endpoints"]

    chat_ids = session["session"]["chat_ids"]
    # tabs = st.tabs(chat_ids)

    # Before running the attack modules, we will get the history first
    # Get History (to update the stream)
    index = 0
    history = get_chat_history()

    # this is a hack for the toxic llm
    if "toxic" in chat_ids[-1]:
        cols = st.columns(len(chat_ids)-1)
    else:
        cols = st.columns(len(chat_ids))

    for col in cols:
        with col:
            st.subheader(f"Chat: {chat_ids[index]}", divider="rainbow")

            chat_history = history[chat_ids[index]]
            for each_turn in chat_history:
                st.chat_message("assistant").write(each_turn[0])
                st.chat_message("user").write(each_turn[1])
        
        index += 1
    
    # for tab in tabs:
    #     with tab:
    #         # get chat history
    #         chat_history = history[chat_ids[index]]
    #         for each_turn in chat_history:
    #             st.chat_message("assistant").write(each_turn[0])
    #             st.chat_message("user").write(each_turn[1])

    #     index += 1
    
    # Prepare to run the attack modules
    endpoints = selected_endpoints
    num_of_prompts = 5 # limit for now

    timestr = time.strftime("%Y%m%d-%H%M%S")

    payload = {
        "name": timestr,
        "recipes": selected_attack_module,
        "endpoints": endpoints,
        "num_of_prompts": num_of_prompts
    }

    results = []

    try:
        response = requests.post(f"{url}/v1/benchmarks?type=recipe",
                                 json=payload)
        recipe_exec_id = response.json()["id"]

        complete = False

        loaded_conversations = []

        # hack for toxic model
        if "toxic" in endpoints[-1]:
            for i in range(len(endpoints)-1):
                loaded_conversations.append([-1])
        else:
            for i in range(len(endpoints)):
                loaded_conversations.append([-1])

        while not complete:
            response = requests.get(f"{url}/v1/benchmarks/status")
            status = response.json()[recipe_exec_id]
            curr_progress = status["curr_progress"]
            error = status["curr_error_messages"]

            if int(curr_progress) == 100:
                complete = True
    
                if error:
                    st.warning("error: {0}".format(error))

                # generate the prompts one by one

                DATABASE = os.path.join(env["DATABASES"], f"{recipe_exec_id}.db")
                import sqlite3
                con = sqlite3.connect(DATABASE)
                cur = con.cursor()

                endpoint_index = 0
                
                if "toxic" in chat_ids[-1]:
                    cols = st.columns(len(chat_ids)-1)
                else:
                    cols = st.columns(len(chat_ids))


                # select the count using the first endpoint
                cur.execute(f'SELECT COUNT(*) FROM {endpoints[0]}')
                row_count = cur.fetchone()[0]

                # show one prompt by one prompt
                for i in range(row_count+1):
                    endpoint_index = 0
                    for col in cols:
                        with col:
                            for row in cur.execute(f'SELECT * FROM {endpoints[endpoint_index]} where id={i};'):
                                st.chat_message("assistant").write(row[5])
                                time.sleep(1) # feel some lag time
                                st.chat_message("user").write(row[6])
                                time.sleep(1) # feel some lag time
                
                        endpoint_index += 1

                st.info(f"We have completed {selected_attack_module}", icon="🥳")
            else:
                with st.spinner(f"Generating new prompts with this automated module... {recipe_exec_id}..."):
                    time.sleep(3)
                


    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
            
    st.chat_input("Type a message here...", key="main_prompt", on_submit=send_prompt)

def view_results(cookbook_exec_id=None):
    """create container based on the endpoints selected
    from the sidebar
    """
    selected_session = st.session_state["selected_session"]
    if selected_session == None or len(selected_session) == 0:
        st.warning("No session is activated.", icon="🔥")
        return
    
    # get the number of chat ID in this session
    session = get_session_info(selected_session)
    selected_endpoints = session["session"]["endpoints"]
    
    st.header(f"Session Name: {selected_session}", divider="rainbow")
    results = get_benchmark_results(cookbook_exec_id)
    final_results = results[0]

    # Create a table here to show the results of the static tests
    tabs = []
    for key in final_results.keys():
        if key not in tabs:
            tabs.append(key)
    
    selected_endpoints_str = ", ".join(x for x in selected_endpoints)
    st.button("Resume Red Teaming", on_click=start_red_teaming)
    
    st.markdown(f"##### Past Results for endpoint: *{selected_endpoints_str}*")
    if len(tabs) > 0:
        created_tabs = st.tabs(tabs)

        tab_index = 0
        for tab in created_tabs:
            this_result = final_results[tabs[tab_index]]

            with tab:
                print_benchmark_results(tabs[tab_index], [this_result, results[1]])

            tab_index += 1
    else:
        st.warning("No predefined tests has been executed yet.")

def start_red_teaming():
    """start a conversational session given the session that is
    activated.
    """    
    selected_session = st.session_state["selected_session"]
    if selected_session == None or len(selected_session) == 0:
        st.warning("No session is activated.", icon="🔥")
        return
    
    # get the number of chat ID in this session
    session = get_session_info(selected_session)
    chat_ids = session["session"]["chat_ids"]

    # hack for toxicity model not to show up
    if "toxic" in chat_ids[-1]:
        cols = st.columns(len(chat_ids)-1)
    else:
        cols = st.columns(len(chat_ids))

    index = 0
    history = get_chat_history()

    for col in cols:
        with col:
            # get chat history
            st.subheader(f"Chat: {chat_ids[index]}", divider="rainbow")
            chat_history = history[chat_ids[index]]
            for each_turn in chat_history:
                st.chat_message("assistant").write(each_turn[0])
                st.chat_message("user").write(each_turn[1])
            
            
        index += 1
            
    st.chat_input("Type a message here...", key="main_prompt", on_submit=send_prompt)

def edit_endpoint():
    selected_endpoint = st.session_state["edit_selected_endpoint"]
    selected_connector = st.session_state["edit_selected_connector"]
    selected_name = st.session_state["edit_endpoint_name"]
    selected_uri = st.session_state["edit_endpoint_uri"]
    selected_token = st.session_state["edit_endpoint_token"]
    selected_mcps = st.session_state["edit_mcps"]
    selected_mc = st.session_state["edit_max_concurrency"]

    # only two mandatory fields
    if not selected_endpoint or not selected_connector or not selected_name:
        st.warning("Cannot add an endpoint", icon="🔥")
        return
    
    try:
        payload = {
            "connector_type": selected_connector,
            "name": selected_name,
            "uri": selected_uri,
            "token": selected_token,
            "max_calls_per_second": selected_mcps,
            "max_concurrency": selected_mc,
            "params": {}
        }

        response = requests.put(f"{url}/v1/llm_endpoints/{selected_endpoint}",
                                 json=payload)
        edit_endpoint = response.json()["message"]
        
        st.success(edit_endpoint)
        return True
    except Exception as e:
        st.warning(e)
        return False

def add_endpoint():
    selected_connector = st.session_state["selected_connector"]
    selected_name = st.session_state["endpoint_name"]
    selected_uri = st.session_state["endpoint_uri"]
    selected_token = st.session_state["endpoint_token"]
    selected_mcps = st.session_state["mcps"]
    selected_mc = st.session_state["max_concurrency"]

    # only two mandatory fields
    if not selected_connector or not selected_name:
        st.warning("Cannot add an endpoint", icon="🔥")
        return

    try:
        payload = {
            "connector_type": selected_connector,
            "name": selected_name,
            "uri": selected_uri,
            "token": selected_token,
            "max_calls_per_second": selected_mcps,
            "max_concurrency": selected_mc,
            "params": {}
        }

        response = requests.post(f"{url}/v1/llm_endpoints",
                                 json=payload)
        add_endpoint = response.json()["message"]

        st.success(add_endpoint)
        return True
    except Exception:
        st.warning("Moonshot WebAPI is not connected.", icon="🔥")
        return False


def run_cookbook():
    selected_session = st.session_state["selected_session"]
    selected_cookbooks = st.session_state["selected_cookbooks"]
    number_of_prompts = st.session_state["number_of_prompts"]

    if selected_session == None or len(selected_session) == 0:
        st.warning("No session is activated.", icon="🔥")
        return

    if len(selected_cookbooks) == 0:
        st.warning("Unable to run. No cookbook is selected.", icon="🔥")
        return
    
    # Retrieve the endpoints
    session = get_session_info(selected_session)
    selected_endpoints = session["session"]["endpoints"]
    
    cookbooks = selected_cookbooks
    endpoints = selected_endpoints
    num_of_prompts = number_of_prompts

    timestr = time.strftime("%Y%m%d-%H%M%S")

    payload = {
        "name": timestr,
        "cookbooks": cookbooks,
        "endpoints": endpoints,
        "num_of_prompts": num_of_prompts
    }

    results = []
    try:
        url = st.session_state["moonshot_fastapi"]
        response = requests.post(f"{url}/v1/benchmarks?type=cookbook",
                                 json=payload)
        
        cookbook_exec_id = response.json()["id"]

        # Check the status
        complete = False
        my_bar = st.progress(0, text="Progressing")

        while not complete:
            response = requests.get(f"{url}/v1/benchmarks/status")
            status = response.json()[cookbook_exec_id]
            curr_progress = status["curr_progress"]
            error = status["curr_error_messages"]

            if int(curr_progress) == 100:
                complete = True
                my_bar.progress(curr_progress, text="Completed.")
                st.info(f"We have completed {cookbook_exec_id}.", icon="🥳")
                my_bar.empty()

                if error:
                    st.warning(error)
            else:
                my_bar.progress(curr_progress, text="Running {0} at {1}%".format(cookbook_exec_id, curr_progress))
                time.sleep(2)
        
        view_results(cookbook_exec_id)

        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        st.warning("{0}".format(e), icon="🔥")
        return results

def create_session():
    """this function will create a session based
    on the selected endpoints
    """
    try:
        endpoints = st.session_state["selected_endpoints"]
        endpoints_str = "_".join(endpoint for endpoint in endpoints)

        url = st.session_state["moonshot_fastapi"]
        timestr = time.strftime("%Y%m%d%H%M%S")
        payload = {
            # "name": "{0}_{1}".format(endpoints_str, timestr),
            "name": "{0}".format(endpoints_str),
            "description": "Session created for {0} on {1}".format(endpoints_str, timestr),
            "endpoints": endpoints,
            "context_strategy": "",
            "prompt_template": ""
        }
        response = requests.post(f"{url}/v1/sessions", json=payload)
            
        status = response.json()

        # Create message boxes for these endpoints
        st.session_state["selected_session"] = status["session"]["session_id"]
        start_red_teaming()

        return status
    except Exception as e:
        st.warning(f"Unable to create a session: {e}")
        return False


############ Pages ############
with st.sidebar:
    # Create a sidebar to store all the configurations
    st.title("Moonshot Lite")

    tabs = st.tabs(["Run", "Endpoint Settings", "Settings"])

    with tabs[-1]:
        st.header("Configure Moonshot Library", divider="rainbow")
        
        st.text_input("FastAPI", key="moonshot_fastapi", value="http://127.0.0.1:5000")
        url = st.session_state["moonshot_fastapi"]
        st.button("Save", key="save")

    with tabs[1]:
        st.header("Create New Endpoint", divider='rainbow')
        
        # Select the Connector First
        options = st.selectbox(
            "Connector",
            list_connector_ids(),
            index=None,
            key="selected_connector",
        )

        # Give it a name
        name_endpoint = st.text_input("Name of the Endpoint*", key="endpoint_name")
        uri = st.text_input("URI", key="endpoint_uri")
        token = st.text_input("Token", type="password", key="endpoint_token")
        
        with st.expander("Advanced Settings"):
            max_calls_per_second = st.number_input("Max API call per second", 60, key="mcps")
            max_concurrency = st.number_input("Max Currency", 2, key="max_concurrency")

        st.button("Add New Endpoint", on_click=add_endpoint, type="primary")

        st.header("Edit Endpoint", divider='rainbow')
        # Select the Connector First
        endpoint = st.selectbox(
            "Endpoint",
            list_endpoints(),
            index=None,
            key="edit_selected_endpoint",
        )

    
        endpoint_info = get_endpoint_info(endpoint)

        if "detail" not in endpoint_info or "Failed" not in endpoint_info["detail"]:
            connector_type = endpoint_info["connector_type"]
            connectors = list_connector_ids()

            options = st.selectbox(
                "Connector",
                connectors,
                index=connectors.index(connector_type),
                key="edit_selected_connector",
            )

            # Give it a name
            name_endpoint = st.text_input("Name of the Endpoint*", endpoint_info["name"], key="edit_endpoint_name")
            uri = st.text_input("URI", endpoint_info["uri"], key="edit_endpoint_uri")
            token = st.text_input("Token", endpoint_info["token"], type="password", key="edit_endpoint_token")
            
            with st.expander("Advanced Settings"):
                max_calls_per_second = st.number_input("Max API call per second", endpoint_info["max_calls_per_second"], key="edit_mcps")
                max_concurrency = st.number_input("Max Currency", endpoint_info["max_concurrency"], key="edit_max_concurrency")

        st.button("Update Endpoint", on_click=edit_endpoint, type="primary")

    with tabs[0]:
        st.header("Existing Test Session", divider='rainbow')
        options = st.selectbox(
            "Current Active Session",
            list_sessions(),
            index=None,
            key="selected_session",
            on_change=start_red_teaming
        )

        with st.popover("Create a new session"):
            options = st.multiselect(
                "Select endpoints",
                list_endpoints(),
                placeholder="",
                key="selected_endpoints"
            )
            st.button("Create New Session", key="create_session", on_click=create_session)

        
        st.header("Automated Red Teaming", divider='rainbow')
        
        redteaming_recipe = list_redteaming_recipes()
        redteaming_recipe_str = ""
        if len(redteaming_recipe) != 0:
            redteaming_recipe_str = redteaming_recipe[0]
        
        options = st.multiselect(
            "Select Attack Modules",
            redteaming_recipe,
            placeholder="Select Attack Modules",
            # default=redteaming_recipe_str,
            key="selected_attack_modules"
        )
        
        st.button("Automate Attack", key="automate_attack", on_click=run_automated_attack)

        st.header("Predefined Tests", divider='rainbow')
        st.button("View Predefined Tests", on_click=print_cookbook_info)

        session = st.session_state["selected_session"]
        endpoints = st.session_state["selected_endpoints"]
        endpoints_str = ", ".join(x for x in endpoints)

        # Create a multiselect dropdown box or users
        # to select cookbook to run
        list_of_cookbooks = list_cookbook()
        if len(list_of_cookbooks) == 0:
            cookbook_str = ""
        else:
            cookbook_str = list_of_cookbooks[0]

        options = st.multiselect(
            "Cookbook",
            list_of_cookbooks,
            # default=[cookbook_str],
            placeholder="Select Cookbooks",
            key="selected_cookbooks"
        )

        number = st.number_input('Number of prompts to run', min_value=0, value=1, key="number_of_prompts")

        col1, col2 = st.columns(2)

        col1.button("Run Tests", on_click=run_cookbook)
        col2.button("View Past Results", on_click=view_results)
        

        ####
        
container = st.container()