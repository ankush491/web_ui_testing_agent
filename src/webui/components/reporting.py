import json
from datetime import datetime

def generate_action_log(history):
    log_entries = []
    if not history:
        return "[]"
    for entry in history.history:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": entry.action.__class__.__name__,
            "details": f"Status: {entry.status}, Output: {entry.output}"
        }
        log_entries.append(log_entry)
    return json.dumps(log_entries, indent=4)

def generate_testing_report(history, task):
    if not history:
        return "No history to generate a report from."
        
    report = f"Objective: {task}\n\n"
    report += "Steps to Reproduce:\n"
    for i, entry in enumerate(history.history):
        report += f"    {i+1}. {entry.action.short_string()}\n"

    report += "\nObserved Issues:\n"
    failures = [entry for entry in history.history if entry.status != 'SUCCESS']
    if not failures:
        report += "No issues observed.\n"
    else:
        for failure in failures:
            report += f"The test encountered a failure during the '{failure.action.short_string()}' step.\n"

    report += "\nDetailed Breakdown of Failures:\n"
    for i, failure in enumerate(failures):
        report += f"    Failure {i+1}: {failure.action.short_string()}\n"
        report += f"        Timestamp: {datetime.now().isoformat()}\n"
        report += f"        Event: {failure.action.__class__.__name__}\n"
        report += f"        Error: {failure.output}\n"

    return report
