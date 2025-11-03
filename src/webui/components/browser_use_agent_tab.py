
import asyncio
import json
import logging
import os
import uuid
from typing import Any, AsyncGenerator, Dict, Optional, Tuple
from datetime import datetime, timezone

import gradio as gr
from browser_use.agent.views import (
    AgentHistoryList,
    AgentOutput,
)
from browser_use.browser.browser import BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.browser.views import BrowserState
from gradio.components import Component
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from src.agent.browser_use.browser_use_agent import BrowserUseAgent
from src.browser.custom_browser import CustomBrowser
from src.controller.custom_controller import CustomController
from src.utils import llm_provider
from src.webui.webui_manager import WebuiManager
from src.utils.report_generation import (
    generate_human_readable_action_log,
    generate_action_log,
    generate_testing_report,
)

logger = logging.getLogger(__name__)


# --- Helper Functions ---


async def _initialize_llm(
    provider: Optional[str],
    model_name: Optional[str],
    temperature: float,
    base_url: Optional[str],
    api_key: Optional[str],
    num_ctx: Optional[int] = None,
) -> Optional[BaseChatModel]:
    """Initializes the LLM based on settings. Returns None if provider/model is missing."""
    if not provider or not model_name:
        logger.info("LLM Provider or Model Name not specified, LLM will be None.")
        return None
    try:
        logger.info(
            f"Initializing LLM: Provider={provider}, Model={model_name}, Temp={temperature}"
        )
        llm = llm_provider.get_llm_model(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            base_url=base_url or None,
            api_key=api_key or None,
            num_ctx=num_ctx if provider == "ollama" else None,
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        gr.Warning(
            f"Failed to initialize LLM '{model_name}' for provider '{provider}'. Please check settings. Error: {e}"
        )
        return None


def _format_agent_output(model_output: AgentOutput) -> str:
    """Formats AgentOutput for display in the chatbot using JSON."""
    content = ""
    if model_output:
        try:
            action_dump = [
                action.model_dump(exclude_none=True) for action in model_output.action
            ]
            state_dump = model_output.current_state.model_dump(exclude_none=True)
            model_output_dump = {
                "current_state": state_dump,
                "action": action_dump,
            }
            json_string = json.dumps(model_output_dump, indent=4, ensure_ascii=False)
            content = f"<pre><code class='language-json'>{json_string}</code></pre>"
        except Exception as e:
            logger.error(f"Error formatting agent output: {e}", exc_info=True)
            content = f"<pre><code>Error formatting agent output.\nRaw output:\n{str(model_output)}</code></pre>"
    return content.strip()


# --- Callback Implementations ---


async def _handle_new_step(
    webui_manager: WebuiManager, state: BrowserState, output: AgentOutput, step_num: int
):
    """Callback to display agent step output and reliably detect failures."""
    if not hasattr(webui_manager, "bu_chat_history"):
        webui_manager.bu_chat_history = []
    step_num -= 1
    logger.info(f"Displaying Step {step_num} in UI.")

    # --- Reliable Failure Detection ---
    # This is the correct place to check for failures, as it's called after every step.
    for action in output.action:
        evaluation = getattr(action, 'evaluation', '')
        if isinstance(evaluation, str) and ("Error:" in evaluation or "Failed to" in evaluation):
            logger.error(f"RELIABLE FAILURE DETECTED at step {step_num}. Terminating agent task.")
            error_message = (
                f"**Critical Error Detected at Step {step_num}:** The test has been automatically stopped due to a failure.\n"
                f"**Reason:** `{evaluation}`"
            )
            webui_manager.bu_chat_history.append({"role": "assistant", "content": error_message})
            
            # Cancel the running task safely
            if webui_manager.bu_current_task and not webui_manager.bu_current_task.done():
                webui_manager.bu_current_task.cancel()
            return # Stop processing this step

    # --- Display Logic ---
    screenshot_html = ""
    screenshot_data = getattr(state, "screenshot", None)
    if screenshot_data:
        try:
            if isinstance(screenshot_data, str) and len(screenshot_data) > 100:
                img_tag = f'''<img src="data:image/jpeg;base64,{screenshot_data}" alt="Step {step_num} Screenshot" style="max-width: 800px; max-height: 600px; object-fit:contain;" />'''
                screenshot_html = img_tag + "<br/>"
            else:
                logger.warning(f"Screenshot for step {step_num} seems invalid.")
                screenshot_html = "**[Invalid screenshot data]**<br/>"
        except Exception as e:
            logger.error(f"Error processing screenshot for step {step_num}: {e}", exc_info=True)
            screenshot_html = "**[Error displaying screenshot]**<br/>"

    formatted_output = _format_agent_output(output)
    step_header = f"--- **Step {step_num}** ---"
    final_content = step_header + "<br/>" + screenshot_html + formatted_output
    chat_message = {"role": "assistant", "content": final_content.strip()}
    webui_manager.bu_chat_history.append(chat_message)
    await asyncio.sleep(0.05)


async def _generate_llm_summary(
    webui_manager: WebuiManager,
    history: AgentHistoryList,
    summary_llm: Optional[BaseChatModel]
):
    """Generates only execution statistics summary."""
    logger.info("Generating execution statistics...")

    errors = history.errors() if callable(getattr(history, 'errors', None)) else []
    has_errors = errors and any(errors)

    summary = (
        "**Execution Statistics**\n"
        f"- Total Actions: {len(history.history)}\n"
        f"- Duration: {history.total_duration_seconds():.2f} seconds\n"
        f"- Total Input Tokens: {history.total_input_tokens()}\n"
        f"- Status: {'Failed' if has_errors else 'Success'}\n"
    )
    
    if history.final_result():
        summary += f"- Final Result: {history.final_result()}\n"

    # Add the summary to chat history
    webui_manager.bu_chat_history.append(
        {"role": "assistant", "content": summary}
    )
    return

    if not summary_llm:
        logger.info("No LLM provided for summary. Using basic summary.")
        webui_manager.bu_chat_history.append(
            {"role": "assistant", "content": fallback_summary}
        )
        return

    try:
        # --- TIMESTAMP CONVERSION TO LOCAL TIME ---
        history_dict = history.model_dump()
        for record in history_dict.get("history", []):
            unix_timestamp = record.get("timestamp")
            if isinstance(unix_timestamp, (int, float)):
                utc_dt = datetime.fromtimestamp(unix_timestamp, timezone.utc)
                local_dt = utc_dt.astimezone()
                record["timestamp"] = local_dt.strftime("%d-%m-%Y. %H.%M")
        
        history_json = json.dumps(history_dict, indent=4)
        # --- END TIMESTAMP CONVERSION ---

        prompt = f'''**Role:** You are an Expert QA Analyst. Your task is to write a professional, accurate, and non-contradictory testing report based **only** on the provided `Execution History (JSON)`.

**Critical Instructions:**
1.  **Single Source of Truth:** The `Execution History (JSON)` is the only truth. Do not invent, infer, or assume any information not present in the log.
2.  **Identify First Failure:** Scan the `history` array in the JSON. An action is considered **failed** if its `evaluation` field contains the string "Error:" or "Failed to". Find the **first** action that failed.
3.  **Stop on Failure:** If a failure is found, your report **must not** include any actions or events that occurred *after* that point of failure. The test is considered terminated at the first error.
4.  **Factual Reporting:** Your narrative and bug report must be based strictly on the actions leading up to and including the first failure.

**Output Format:**
You **must** generate two separate and distinct Markdown sections: "Action Log" and "Testing Report".

---

### Action Log

Create a human-readable, bulleted list of actions performed **up to the point of the first failure**. If there is no failure, log all actions. Each item must follow this exact format:
- `[timestamp]` A clear description of the action. (Outcome: Success/Failure)

**Guidelines for Action Log:**
- The `timestamp` field in the JSON is already formatted as `DD-MM-YYYY. HH.MM`. **Use it directly.**
- **Translate technical details into plain English.** Instead of "Input 'Sagar123#' into index 3", write "Entered 'Sagar123#' into the 'Password' field." Use the `aria_label`, `placeholder`, or `text` from the JSON to identify the element.
- Be concise and factual. The outcome must reflect the `evaluation` field.

---

### Testing Report

#### 1. Objective
(Write a single sentence describing the test's goal, based on the `objective` field.)

#### 2. Test Summary
(Write a short narrative paragraph summarizing the user journey. If the test failed, clearly state at which step it was terminated and why, based on the first failed action.)

#### 3. Identified Bugs
(If no failures were identified, write "No bugs were identified during this test." If a failure was identified, create a structured bug report for the **first failed action only**.)

**Bug Title:** (A brief, clear title for the bug, e.g., "Login Fails Due to Page Navigation Error")

**Description:** (A step-by-step description of how to reproduce the bug based on the action log leading to the failure.)
1.  ...
2.  ...

**User Impact:** (Explain the consequence of this bug for the user, e.g., "The user is completely blocked from logging into the application.")

**Error Message:**
```
(Provide the exact, verbatim error message from the `evaluation` field of the first failed action.)
```

---

**Execution History (JSON):**
```json
{history_json}
```
'''
        response = await summary_llm.ainvoke(prompt)
        llm_summary = response.content
        logger.info("Successfully generated professional QA summary.")
    except Exception as e:
        logger.error(f"Failed to generate professional QA summary: {e}", exc_info=True)
        gr.Warning("Failed to generate a detailed report with the LLM. Displaying a basic summary instead.")
        llm_summary = fallback_summary

    webui_manager.bu_chat_history.append(
        {"role": "assistant", "content": llm_summary}
    )


async def _ask_assistant_callback(
    webui_manager: WebuiManager, query: str, browser_context: BrowserContext
) -> Dict[str, Any]:
    """Callback triggered by the agent's ask_for_assistant action."""
    logger.info("Agent requires assistance. Waiting for user input.")
    webui_manager.bu_chat_history.append(
        {
            "role": "assistant",
            "content": f"**Need Help:** {query}\nPlease provide information or perform the required action in the browser, then type your response/confirmation below and click 'Submit Response'.",
        }
    )
    webui_manager.bu_response_event = asyncio.Event()
    webui_manager.bu_user_help_response = None
    try:
        await asyncio.wait_for(webui_manager.bu_response_event.wait(), timeout=3600.0)
    except asyncio.TimeoutError:
        logger.warning("Timeout waiting for user assistance.")
        webui_manager.bu_chat_history.append(
            {"role": "assistant", "content": "**Timeout:** No response received. Trying to proceed."}
        )
        return {"response": "Timeout: User did not respond."}
    finally:
        webui_manager.bu_response_event = None

    response = webui_manager.bu_user_help_response
    webui_manager.bu_chat_history.append({"role": "user", "content": response})
    return {"response": response}


# --- Core Agent Execution Logic ---


async def run_agent_task(
    webui_manager: WebuiManager, components: Dict[gr.components.Component, Any]
) -> AsyncGenerator[Dict[gr.components.Component, Any], None]:
    """Handles the entire lifecycle of initializing and running the agent."""
    # --- Component Access ---
    user_input_comp = webui_manager.get_component_by_id("browser_use_agent.user_input")
    run_button_comp = webui_manager.get_component_by_id("browser_use_agent.run_button")
    stop_button_comp = webui_manager.get_component_by_id("browser_use_agent.stop_button")
    pause_resume_button_comp = webui_manager.get_component_by_id("browser_use_agent.pause_resume_button")
    clear_button_comp = webui_manager.get_component_by_id("browser_use_agent.clear_button")
    chatbot_comp = webui_manager.get_component_by_id("browser_use_agent.chatbot")
    history_file_comp = webui_manager.get_component_by_id("browser_use_agent.agent_history_file")
    gif_comp = webui_manager.get_component_by_id("browser_use_agent.recording_gif")
    browser_view_comp = webui_manager.get_component_by_id("browser_use_agent.browser_view")


    task = components.get(user_input_comp, "").strip()
    if not task:
        gr.Warning("Please enter a task.")
        yield {run_button_comp: gr.update(interactive=True)}
        return

    # --- UI Updates & Settings Initialization ---
    webui_manager.bu_chat_history.append({"role": "user", "content": task})
    yield {
        user_input_comp: gr.Textbox(value="", interactive=False, placeholder="Agent is running..."),
        run_button_comp: gr.Button(value="⏳ Running...", interactive=False),
        stop_button_comp: gr.Button(interactive=True),
        pause_resume_button_comp: gr.Button(value="⏸️ Pause", interactive=True, visible=False),
        clear_button_comp: gr.Button(interactive=False),
        chatbot_comp: gr.update(value=webui_manager.bu_chat_history),
        history_file_comp: gr.update(value=None),
        gif_comp: gr.update(value=None, visible=False),
        browser_view_comp: gr.update(visible=False),
    }

    def get_setting(key, default=None):
        comp = webui_manager.id_to_component.get(f"agent_settings.{key}")
        return components.get(comp, default) if comp else default

    llm_provider_name = get_setting("llm_provider", None)
    llm_model_name = get_setting("llm_model_name", None)
    llm_temperature = get_setting("llm_temperature", 0.6)
    use_vision = get_setting("use_vision", True)
    ollama_num_ctx = get_setting("ollama_num_ctx", 16000)
    llm_base_url = get_setting("llm_base_url") or None
    llm_api_key = get_setting("llm_api_key") or None
    max_steps = get_setting("max_steps", 100)
    mcp_server_config_str = components.get(webui_manager.id_to_component.get("agent_settings.mcp_server_config"))
    mcp_server_config = json.loads(mcp_server_config_str) if mcp_server_config_str else None

    def get_browser_setting(key, default=None):
        comp = webui_manager.id_to_component.get(f"browser_settings.{key}")
        return components.get(comp, default) if comp else default

    headless = get_browser_setting("headless", False)
    keep_browser_open = get_browser_setting("keep_browser_open", False)
    window_w = int(get_browser_setting("window_w", 1280))
    window_h = int(get_browser_setting("window_h", 1100))
    save_agent_history_path = get_browser_setting("save_agent_history_path", "./tmp/agent_history")
    os.makedirs(save_agent_history_path, exist_ok=True)

    main_llm = await _initialize_llm(
        llm_provider_name, llm_model_name, llm_temperature, llm_base_url, llm_api_key,
        ollama_num_ctx if llm_provider_name == "ollama" else None
    )

    async def ask_callback_wrapper(query: str, browser_context: BrowserContext) -> Dict[str, Any]:
        return await _ask_assistant_callback(webui_manager, query, browser_context)

    if not webui_manager.bu_controller:
        webui_manager.bu_controller = CustomController(ask_assistant_callback=ask_callback_wrapper)
        await webui_manager.bu_controller.setup_mcp_client(mcp_server_config)

    agent_task = None
    agent_history = None
    history_file = None
    try:
        if not keep_browser_open or not webui_manager.bu_browser:
            if webui_manager.bu_browser: await webui_manager.bu_browser.close()
            webui_manager.bu_browser = CustomBrowser(config=BrowserConfig(headless=headless, window_width=window_w, window_height=window_h))
            webui_manager.bu_browser_context = await webui_manager.bu_browser.new_context()

        webui_manager.bu_agent_task_id = str(uuid.uuid4())
        history_file = os.path.join(save_agent_history_path, f"{webui_manager.bu_agent_task_id}.json")

        async def step_callback_wrapper(state: BrowserState, output: AgentOutput, step_num: int):
            await _handle_new_step(webui_manager, state, output, step_num)

        webui_manager.bu_agent = BrowserUseAgent(
            task=task, llm=main_llm, browser=webui_manager.bu_browser,
            browser_context=webui_manager.bu_browser_context, controller=webui_manager.bu_controller,
            register_new_step_callback=step_callback_wrapper, use_vision=use_vision,
            source="webui"
        )
        webui_manager.bu_agent.state.agent_id = webui_manager.bu_agent_task_id

        agent_run_coro = webui_manager.bu_agent.run(max_steps=max_steps)
        agent_task = asyncio.create_task(agent_run_coro)
        webui_manager.bu_current_task = agent_task

        # --- Simplified UI Streaming Loop ---
        last_chat_len = len(webui_manager.bu_chat_history)
        while not agent_task.done():
            if len(webui_manager.bu_chat_history) > last_chat_len:
                yield {chatbot_comp: gr.update(value=webui_manager.bu_chat_history)}
                last_chat_len = len(webui_manager.bu_chat_history)
            await asyncio.sleep(0.1)

        agent_history = await agent_task

    except (Exception, asyncio.CancelledError) as e:
        if isinstance(e, asyncio.CancelledError):
            logger.warning("Agent task was cancelled. This is expected if a failure was detected or stop was clicked.")
            if webui_manager.bu_agent:
                agent_history = webui_manager.bu_agent.state.history
        else:
            logger.error(f"Error during agent execution: {e}", exc_info=True)
            error_message = f"**Agent Execution Error:**\n```\n{type(e).__name__}: {e}\n```"
            webui_manager.bu_chat_history.append({"role": "assistant", "content": error_message})
            gr.Error(f"Agent execution failed: {e}")

    finally:
        # This block now provides a single, consolidated UI update at the end.
        final_updates = {
            user_input_comp: gr.update(value="", interactive=True, placeholder="Enter your next task..."),
            run_button_comp: gr.update(value="▶️ Submit Task", interactive=True),
            stop_button_comp: gr.update(value="⏹️ Stop", interactive=False),
            pause_resume_button_comp: gr.update(value="⏸️ Pause", interactive=False, visible=False),
            clear_button_comp: gr.update(interactive=True),
        }

        if agent_history:
            # Generate deterministic, local reports first (using system timestamps)
            try:
                human_log = generate_human_readable_action_log(agent_history)
                testing_report = generate_testing_report(task, agent_history)
                # Append both to the assistant chat history as separate messages so the user sees accurate logs
                webui_manager.bu_chat_history.append({"role": "assistant", "content": "### Action Log\n" + human_log})
                webui_manager.bu_chat_history.append({"role": "assistant", "content": "### Testing Report\n" + testing_report})
            except Exception as e:
                logger.error(f"Failed to generate local reports: {e}", exc_info=True)

            # Still call the LLM summary if available
            await _generate_llm_summary(webui_manager, agent_history, main_llm)
            if history_file:
                with open(history_file, "w") as f:
                    f.write(agent_history.model_dump_json(indent=4))
                final_updates[history_file_comp] = gr.File(value=history_file)
        
        # Always ensure the chatbot is updated with the final history
        final_updates[chatbot_comp] = gr.update(value=webui_manager.bu_chat_history)

        if not keep_browser_open:
            if webui_manager.bu_browser:
                await webui_manager.bu_browser.close()
            webui_manager.bu_browser = None
            webui_manager.bu_browser_context = None

        webui_manager.bu_current_task = None
        yield final_updates


# --- Button Click Handlers ---


async def handle_submit(
    webui_manager: WebuiManager, components: Dict[gr.components.Component, Any]
):
    """Handles clicks on the main 'Submit' button by running the agent task."""
    async for update in run_agent_task(webui_manager, components):
        yield update


def handle_stop(webui_manager: WebuiManager) -> Dict[gr.Component, Any]:
    """Handles clicks on the 'Stop' button. Now synchronous."""
    if webui_manager.bu_current_task and not webui_manager.bu_current_task.done():
        webui_manager.bu_current_task.cancel()
        logger.info("Manual stop requested. Cancelling agent task.")
    
    stop_button_comp = webui_manager.get_component_by_id("browser_use_agent.stop_button")
    return {
        stop_button_comp: gr.update(interactive=False, value="⏹️ Stopping..."),
    }


def handle_clear(webui_manager: WebuiManager) -> Tuple[Dict, str, None, None, gr.Button, gr.Button, gr.Button, gr.Button, gr.HTML]:
    """
    Handles clicks on the 'Clear' button. Now synchronous and returns a tuple for all outputs.
    This resolves the critical ValueError crash.
    """
    if webui_manager.bu_current_task and not webui_manager.bu_current_task.done():
        webui_manager.bu_current_task.cancel()
        logger.info("Clear requested. Cancelling agent task if running.")
    
    webui_manager.init_browser_use_agent() # Resets bu_chat_history
    
    # Returns a tuple of updates for all 9 components linked to the clear button.
    return (
        [], # chatbot
        "", # user_input
        None, # agent_history_file
        None, # recording_gif
        gr.update(interactive=True, value="▶️ Submit Task"), # run_button
        gr.update(interactive=False, value="⏹️ Stop"), # stop_button
        gr.update(interactive=False, value="⏸️ Pause", visible=False), # pause_resume_button
        gr.update(interactive=True), # clear_button
        gr.update(visible=False), # browser_view
    )


# --- Tab Creation Function ---

def create_browser_use_agent_tab(webui_manager: WebuiManager):
    """Create the run agent tab, defining UI, state, and handlers."""
    webui_manager.init_browser_use_agent()

    with gr.Column():
        chatbot = gr.Chatbot(
            lambda: webui_manager.bu_chat_history,
            elem_id="browser_use_chatbot",
            label="Agent Interaction",
            type="messages",
            height=600,
            show_copy_button=True,
        )
        user_input = gr.Textbox(
            label="Your Task or Response",
            placeholder="Enter your task here or provide assistance when asked.",
            lines=3,
            interactive=True,
        )
        with gr.Row():
            stop_button = gr.Button("⏹️ Stop", interactive=False, variant="stop")
            pause_resume_button = gr.Button("⏸️ Pause", interactive=False, visible=False)
            clear_button = gr.Button("🗑️ Clear", interactive=True)
            run_button = gr.Button("▶️ Submit Task", variant="primary")

        browser_view = gr.HTML(visible=False)
        with gr.Column():
            agent_history_file = gr.File(label="Agent History JSON", interactive=False)
            recording_gif = gr.Image(label="Task Recording GIF", interactive=False, visible=False)

    tab_components = {
        "chatbot": chatbot, "user_input": user_input, "run_button": run_button,
        "stop_button": stop_button, "pause_resume_button": pause_resume_button,
        "clear_button": clear_button, "agent_history_file": agent_history_file,
        "recording_gif": recording_gif, "browser_view": browser_view,
    }
    webui_manager.add_components("browser_use_agent", tab_components)

    all_managed_components = webui_manager.get_components()
    
    # The outputs for run/submit must match the order of `tab_components`
    run_tab_outputs = list(tab_components.values())

    async def submit_wrapper(*args):
        component_dict = dict(zip(all_managed_components, args))
        async for update in handle_submit(webui_manager, component_dict):
            yield update

    run_button.click(
        submit_wrapper, 
        inputs=all_managed_components, 
        outputs=run_tab_outputs
    )
    user_input.submit(
        submit_wrapper, 
        inputs=all_managed_components, 
        outputs=run_tab_outputs
    )

    stop_button.click(
        lambda: handle_stop(webui_manager), 
        inputs=None, 
        outputs=[stop_button]
    )
    
    # The outputs must be a list of components in the exact order expected by handle_clear's tuple
    clear_outputs = [
        chatbot, user_input, agent_history_file, recording_gif, 
        run_button, stop_button, pause_resume_button, clear_button, browser_view
    ]
    clear_button.click(
        lambda: handle_clear(webui_manager), 
        inputs=None, 
        outputs=clear_outputs
    )
