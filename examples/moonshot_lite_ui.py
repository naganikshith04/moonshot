import streamlit as st
import pandas as pd
import sys, os, json, time
sys.path.insert(0, '../')
import requests

from moonshot.api import (
    api_create_recipe,
    api_create_cookbook,
    api_create_endpoint,
    # api_create_recipe_executor,
    # api_create_cookbook_executor,
    api_create_session,
    api_get_session,
    api_get_all_connector_type,
    api_get_all_endpoint,
    api_get_all_cookbook,
    api_get_all_recipe,
    # api_get_all_executor,
    api_get_all_session_detail,
    api_get_all_prompt_template_detail,
    api_get_all_context_strategy_name,
    api_get_session_chats_by_session_id,
    # api_load_executor,
    api_set_environment_variables,
    api_send_prompt,
    api_update_context_strategy,
    api_update_prompt_template,
)

moonshot_path = "../moonshot/data/"

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
    url = st.session_state["moonshot_fastapi"]
    try:
        response = requests.get(f"{url}/v1/llm_endpoints/")
        get_endpoints = response.json()
        for endpoint in get_endpoints:
            results.append(endpoint["id"])
            
        return results
    except Exception:
        st.warning("Moonshot WebAPI is not connected.", icon="🔥")
        return results

def list_cookbook():
    """Get a list of cookbooks and returns the ID
    as a list
    """
    results = []
    # get all cookbook
    url = st.session_state["moonshot_fastapi"]
    try:
        response = requests.get(f"{url}/v1/cookbooks/")
        get_cookbooks = response.json()
        for cookbook in get_cookbooks:
            results.append(cookbook["id"])
        return results
    except Exception:
        st.warning("Moonshot WebAPI is not connected.", icon="🔥")
        return results

def list_results(cookbook_ids):
    """Get a list of benchmark results and return its ID and endpoints as a pair
    """
    results = {}
    url = st.session_state["moonshot_fastapi"]
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
    """Get a list of sessions and returns the ID
    as a list
    """
    results = []
    # get all cookbook
    url = st.session_state["moonshot_fastapi"]
    try:
        response = requests.get(f"{url}/v1/sessions/")
        get_sessions = response.json()
        for session in get_sessions:
            results.append(session["session_id"])
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
        st.warning("No session is activated.")
        return
    
    session = get_session_info(selected_session)
    selected_endpoints = session["session"]["endpoints"]

    total_match = 0

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
                                for prompt_template in dataset["prompt_templates"]:
                                    real_results = prompt_template["metrics"]
                                    if cookbook_id in final_results:
                                        final_results[cookbook_id].append((cookbook_name, 
                                                                            recipe_id, 
                                                                            real_results, 
                                                                            model_id))
                                    else:
                                        print("adding {0} {1} {2}".format(cookbook_name, recipe_id, model_id))
                                        final_results[cookbook_id] = [(cookbook_name, 
                                                                        recipe_id, 
                                                                        real_results, 
                                                                        model_id)]
                total_match += 1

                if total_match >= limit:
                    return final_results

    return final_results

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

def print_benchmark_results(cookbook_id, results):
    exec_info = get_exec_info_by_id(cookbook_id).json()
    result_name = os.path.join(env["RESULTS"], exec_info["metadata"]["id"] + ".json")
    
    with open(result_name, "r") as f:
        data = json.load(f)

    cookbooks = {}
    metrics_str = []
    start_time = data["metadata"]["start_time"]
    end_time = data["metadata"]["end_time"]
    duration = data["metadata"]["duration"]
    num_of_prompts = data["metadata"]["num_of_prompts"]

    metadata_table = "| Start Time | End Time | Duration | Number of Prompts Per Recipe | "
    metadata_table += "\n| :------------ | :------------ | :------------  | :------------ |"
    metadata_table += f"\n| {start_time} | {end_time} | {duration}  | {num_of_prompts} |"
    st.markdown(metadata_table)
    st.divider()
    
    table_columns = []
    table_column = ""

    for result in results:
        cookbook = result[0]

        if cookbook not in cookbooks.keys():
            cookbooks[cookbook] = [[result[1], result[2], result[3]]]
        else:
            cookbooks[cookbook].append([result[1], result[2], result[3]])
    
    for cookbook in cookbooks:  
        # create a new table
        st.markdown(f"### {cookbook}")
        table_column = "| Endpoint | Recipe  | Metrics |"
        table_column += "\n| :------------| :------------ | :------------ |"    
        
        for r in cookbooks[cookbook]:    
            if len(r) == 3:
                model_id = r[2]
                recipe_name = r[0]
                metrics = r[1]
                table_column += "\n| {0} | {1} | {2} |".format(model_id, recipe_name, metrics)
        st.markdown(table_column, unsafe_allow_html=True)
        st.divider()
        
    st.divider()

    # Expand with the details of the results
    with st.expander("See details of the results"):
        st.write(data)

def send_prompt():
    selected_session = st.session_state["selected_session"]
    if selected_session == None or len(selected_session) == 0:
        st.warning("No session is activated.")
        return
    
    # get the number of chat ID in this session
    session = get_session_info(selected_session)
    chat_ids = session["session"]["chat_ids"]
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
def create_message_box(cookbook_exec_id=None):
    """create container based on the endpoints selected
    from the sidebar
    """
    selected_session = st.session_state["selected_session"]
    if selected_session == None or len(selected_session) == 0:
        st.warning("No session is activated.")
        return
    
    # get the number of chat ID in this session
    session = get_session_info(selected_session)
    selected_endpoints = session["session"]["endpoints"]
    
    st.header(f"Session {selected_session}", divider="rainbow")
    final_results = get_benchmark_results(cookbook_exec_id)

    # Create a table here to show the results of the static tests
    tabs = []
    for key in final_results.keys():
        if key not in tabs:
            tabs.append(key)
    
    selected_endpoints_str = ", ".join(x for x in selected_endpoints)
    st.markdown(f"##### Past Results for endpoint: *{selected_endpoints_str}*")
    
    if len(tabs) > 0:
        created_tabs = st.tabs(tabs)

        tab_index = 0
        for tab in created_tabs:
            results = final_results[tabs[tab_index]]

            with tab:
                print_benchmark_results(tabs[tab_index], results)

            tab_index += 1
    else:
        st.warning("No predefined tests has been executed yet.")

def start_red_teaming():
    """start a conversational session given the session that is
    activated.
    """    
    selected_session = st.session_state["selected_session"]
    if selected_session == None or len(selected_session) == 0:
        st.warning("No session is activated.")
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
        st.warning("Unable to run. No endpoint is selected.", icon="🔥")
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
        
        create_message_box(cookbook_exec_id)

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

    tabs = st.tabs(["View", "Run", "Settings"])

    with tabs[-1]:
        st.header("Configure Moonshot Library", divider="rainbow")
        
        st.text_input("FastAPI", key="moonshot_fastapi", value="http://127.0.0.1:5000")
        st.button("Save", key="save")

    with tabs[0]:
        st.header("Existing Test Session", divider='rainbow')
        options = st.selectbox(
            "Current Active Session",
            list_sessions(),
            index=None,
            key="selected_session"
        )

        col1, col2 = st.columns(2)

        button1 = col2.button("View Predefined Results", on_click=create_message_box, type="primary")
        button2 = col1.button("Start Testing", on_click=start_red_teaming, type="primary")

        st.header("Create New Session", divider='rainbow')
        options = st.multiselect(
            "Select endpoints",
            list_endpoints(),
            placeholder="",
            key="selected_endpoints"
        )

        st.button("Create New Session", type="primary", key="create_session", on_click=create_session)

    with tabs[1]:
        session = st.session_state["selected_session"]
        endpoints = st.session_state["selected_endpoints"]
        endpoints_str = ", ".join(x for x in endpoints)

        if endpoints_str == "":
            endpoints_str = None

        st.markdown(f"You are currently running tests on **{session}**")
        st.header("Predefined Tests", divider='rainbow')

        # Create a multiselect dropdown box or users
        # to select cookbook to run
        options = st.multiselect(
            "Cookbook",
            list_cookbook(),
            default=[list_cookbook()[0]],
            placeholder="Select Cookbooks",
            key="selected_cookbooks"
        )

        number = st.number_input('Number of prompts to run', min_value=0, value=1, key="number_of_prompts")
        st.button("Run Predefined Test(s)", type="primary", on_click=run_cookbook)

        ####
        st.header("Attack Modules", divider='rainbow')
        # This is not available...
        # options = st.multiselect(
        #     "Select Attack Modules",
        #     list_cookbook(),
        #     placeholder="Select Cookbooks",
        #     key="selected_cookbooks"
        # )


container = st.container()