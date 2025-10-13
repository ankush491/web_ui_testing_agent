import gradio as gr
from src.webui.webui_manager import WebuiManager

def create_action_log_tab(webui_manager: WebuiManager):
    with gr.Group():
        action_log = gr.Textbox(
            label="Action Log",
            lines=20,
            interactive=False
        )
        testing_report = gr.Textbox(
            label="Testing Report",
            lines=20,
            interactive=False
        )

    tab_components = {
        "action_log": action_log,
        "testing_report": testing_report,
    }
    webui_manager.add_components("action_log_and_report", tab_components)