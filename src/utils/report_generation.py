import json
from datetime import datetime
from browser_use.agent.views import AgentHistoryList, AgentOutput

def format_action_log_entry(output: AgentOutput, success: bool, error_message: str = "") -> str:
    """Formats a single action log entry."""
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    if not output.action:
        return ""
        
    action = output.action[0]
    event = action.action_name
    details = ""

    if success:
        if event == "navigate_to":
            details = f"Successfully navigated to {action.args.get('url')}"
        elif event == "type_text":
            details = f"Successfully entered '{action.args.get('text')}' in the '{action.args.get('label')}' field."
        elif event == "click":
            details = f"Successfully clicked on the '{action.args.get('label')}' button."
        else:
            details = f"Successfully executed {event} with args {action.args}"
    else:
        if event == "navigate_to":
            details = f"Failed to navigate to {action.args.get('url')}. Error: {error_message}"
        elif event == "type_text":
            details = f"Failed to enter '{action.args.get('text')}' in the '{action.args.get('label')}' field. Error: {error_message}"
        elif event == "click":
            details = f"Failed to click on the '{action.args.get('label')}' button. Error: {error_message}"
        else:
            details = f"FAILURE: {event} with args {action.args}. Error: {error_message}"
            
    log_entry = {
        "timestamp": timestamp,
        "event": event,
        "details": details
    }
    
    return json.dumps(log_entry)

def generate_action_log(history: AgentHistoryList) -> str:
    """Generates the full action log from agent history."""
    log_entries = []
    errors = history.errors()
    for i, entry in enumerate(history.history):
        if entry.output and entry.output.action:
            # Check if there was an error at this step
            is_success = not (errors and i < len(errors) and errors[i])
            err_msg = str(errors[i]) if not is_success else ""
            
            # format_action_log_entry returns a JSON string
            log_entry_str = format_action_log_entry(entry.output, is_success, err_msg)
            if log_entry_str:
                log_entries.append(log_entry_str)
    
    return "\n".join(log_entries)

def generate_testing_report(task: str, history: AgentHistoryList) -> str:
    """Generates the final testing report."""
    
    objective = f"Objective: To test the following task: {task}"
    
    steps_to_reproduce = "Steps to Reproduce:\n"
    for i, entry in enumerate(history.history):
        if entry.output and entry.output.action:
            action = entry.output.action[0]
            steps_to_reproduce += f"{i+1}. {action.action_name} with args: {action.args}\n"
            
    observed_issues = "Observed Issues:\n"
    errors = history.errors()
    if errors and any(errors):
        observed_issues += "The test encountered one or more failures. See the detailed breakdown below.\n"
    else:
        observed_issues += "The test completed successfully with no observed issues.\n"

    detailed_breakdown = "Detailed Breakdown of Failures:\n"
    if errors and any(errors):
        for i, error in enumerate(errors):
            if error:
                entry = history.history[i]
                action = entry.output.action[0] if entry.output and entry.output.action else None
                action_str = f"{action.action_name} with args: {action.args}" if action else "N/A"
                
                detailed_breakdown += f"Failure {i+1}: {action_str}\n"
                detailed_breakdown += f"Timestamp: {entry.start_time.isoformat() if entry.start_time else 'N/A'}\n"
                detailed_breakdown += f"Error: {error}\n\n"
    else:
        detailed_breakdown += "No failures were recorded.\n"

    return f"{objective}\n\n{steps_to_reproduce}\n\n{observed_issues}\n\n{detailed_breakdown}"