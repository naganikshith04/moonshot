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

                print(real_metric)
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
    tabs = st.tabs(chat_ids)

    # Before running the attack modules, we will get the history first
    # Get History (to update the stream)
    index = 0
    history = get_chat_history()

    for tab in tabs:
        with tab:
            # get chat history
            chat_history = history[chat_ids[index]]
            for each_turn in chat_history:
                st.chat_message("assistant").write(each_turn[0])
                st.chat_message("user").write(each_turn[1])

        index += 1
    
    # Prepare to run the attack modules
    endpoints = selected_endpoints
    num_of_prompts = 1

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
        
        while not complete:
            response = requests.get(f"{url}/v1/benchmarks/status")
            print(response.json())
            return
            status = response.json()[recipe_exec_id]
            curr_progress = status["curr_progress"]
            error = status["curr_error_messages"]

            if int(curr_progress) == 100:
                complete = True
                st.info(f"We have completed {selected_attack_module}", icon="🥳")

                if error:
                    st.warning(error)
            else:
                # can I retrieve the partial results here?
                time.sleep(2)

    except Exception as e:
        st.warning(e)

    # Sleep
    time.sleep(1)

    
            
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

    tabs = st.tabs(chat_ids)

    index = 0
    history = get_chat_history()

    for tab in tabs:
        with tab:
            # get chat history
            chat_history = history[chat_ids[index]]
            for each_turn in chat_history:
                print(each_turn)
                st.chat_message("assistant").write(each_turn[0])
                st.chat_message("user").write(each_turn[1])
            
            
        index += 1
            
    st.chat_input("Type a message here...", key="main_prompt", on_submit=send_prompt)


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

    tabs = st.tabs(["Run", "Settings"])

    with tabs[-1]:
        st.header("Configure Moonshot Library", divider="rainbow")
        
        st.text_input("FastAPI", key="moonshot_fastapi", value="http://127.0.0.1:5000")
        url = st.session_state["moonshot_fastapi"]
        st.button("Save", key="save")

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
            placeholder="",
            default=redteaming_recipe_str,
            key="selected_attack_modules"
        )
        
        st.button("Automate Attacks", key="automate_attack", on_click=run_automated_attack)

        st.header("Predefined Tests", divider='rainbow')
        
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
            default=[cookbook_str],
            placeholder="Select Cookbooks",
            key="selected_cookbooks"
        )

        number = st.number_input('Number of prompts to run', min_value=0, value=1, key="number_of_prompts")

        col1, col2 = st.columns(2)

        col1.button("Run Tests", on_click=run_cookbook)
        col2.button("View Past Results", on_click=view_results)

        ####
        
container = st.container()