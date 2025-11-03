"""Deterministic report generation helpers.

This module provides classes and functions for generating structured action logs and test reports.
The action log tracks all actions with timestamps in machine-readable format,
while the test report provides a human-readable summary of test execution.

Formats:
- Action Log: JSON array of entries with timestamp, event, and details
  Example: {"timestamp": "2025-09-15T10:49:45.679Z", "event": "Navigation", 
           "details": "Successfully navigated to the login page"}
           
- Testing Report: Structured text with sections for:
  - Objective
  - Steps to Reproduce
  - Observed Issues
  - Detailed Breakdown of Failures

Timestamps:
- Machine timestamp: UTC ISO-8601 with milliseconds and trailing Z
  (e.g. 2025-09-15T10:49:45.679Z)
- Human timestamp: DD-MM-YYYY. HH.MM (local timezone)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utc_iso_ms_now() -> str:
    """Return current UTC timestamp in ISO format with milliseconds."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _format_human_timestamp_from_unix(ts: Any) -> str:
    """Convert unix timestamp to human-readable format."""
    try:
        return datetime.fromtimestamp(float(ts)).astimezone().strftime("%d-%m-%Y. %H.%M")
    except Exception:
        return datetime.now().astimezone().strftime("%d-%m-%Y. %H.%M")


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """Safely get attribute or dictionary value from an object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_steps(history: Any) -> List[Any]:
    """Get steps from history object or list."""
    if history is None:
        return []
    if isinstance(history, list):
        return history
    return _safe_get(history, "history", []) or []


def _get_errors(history: Any) -> List[Any]:
    """Get errors from history object."""
    e = _safe_get(history, "errors", None)
    if callable(e):
        try:
            result = e()
            return result if isinstance(result, list) else []
        except Exception:
            return []
    return e if isinstance(e, list) else []


class ActionLogger:
    """Handles logging of actions with timestamps during test execution."""

    def __init__(self):
        self.actions: List[Dict[str, Any]] = []

    def _get_element_description(self, args: Dict[str, Any]) -> str:
        """Get a human-readable description of an element."""
        # Get all possible identifiers
        label = (
            args.get('label') or 
            args.get('text') or 
            args.get('aria_label') or
            args.get('button') or
            args.get('link') or
            args.get('element')
        )
        
        if not label:
            return "element"
            
        # Clean up the label
        label = str(label).strip().strip("'\"")
        
        # Determine if it's a special type of element
        if any(x in str(args).lower() for x in ["button", "submit", "btn"]):
            return f"the '{label}' button"
        elif "link" in str(args).lower():
            return f"the '{label}' link"
        elif "refresh" in label.lower():
            return f"the refresh {label}"
        elif "icon" in label.lower():
            return f"the {label}"
        else:
            return f"'{label}'"

    def _format_action_description(self, action_name: str, args: Dict[str, Any]) -> str:
        """Generate a human-readable description of the action."""
        action = str(action_name).lower().strip()
        
        # Navigation
        if "navigate" in action or "goto" in action:
            url = args.get('url', 'unknown URL')
            return f"Navigated to {url}"
            
        # Text Input    
        if "type" in action or "input" in action or "enter" in action:
            text = args.get('text', '')
            field = (args.get('label') or args.get('field') or 
                    args.get('input') or args.get('selector', '')).strip("'\"")
            return f"Entered '{text}' into the '{field}' field"
            
        # Clicking
        if "click" in action or "press" in action:
            element = (args.get('button') or args.get('link') or 
                      args.get('text') or args.get('label', '')).strip("'\"")
            if "submit" in str(element).lower():
                return f"Clicked the Submit button"
            elif element:
                return f"Clicked on '{element}'"
            return "Clicked element"
            
        # Selection
        if "select" in action:
            value = args.get('value', '')
            field = args.get('label', '').strip("'\"")
            return f"Selected '{value}' from '{field}'"
            
        # Waiting
        if "wait" in action:
            text = args.get('text', '')
            if text:
                return f"Waited for '{text}' to appear"
            return "Waited for element"
            
        # Form submission
        if "submit" in action:
            return "Submitted the form"
            
        # Browser control
        if "close" in action:
            what = args.get('what', 'browser')
            return f"Closed {what}"
        if "refresh" in action:
            return "Refreshed the page"
            
        # If we have an action but no specific format
        if action:
            action = action.replace("_", " ").title()
            if args:
                details = [f"{k}='{v}'" for k, v in args.items() if v]
                if details:
                    return f"{action} ({', '.join(details)})"
            return action
            
        # Final fallback
        return "Performed unknown action"
            
        # If we have an action but no specific format, make it readable
        if action:
            action = action.replace("_", " ").title()
            # Include args if available
            if args:
                details = [f"{k}='{v}'" for k, v in args.items() if v]
                if details:
                    return f"{action} ({', '.join(details)})"
            return action
            
        # Final fallback
        return "Performed browser action"


    def log_action(self, step_data: Any, success: bool = True, error_msg: str = "") -> None:
        """Log an action with its original timestamp and details.

        Args:
            step_data: The step data containing action details and metadata
            success: Whether the action succeeded or failed
            error_msg: Error message if the action failed
        """
        mo = _safe_get(step_data, "model_output", None) or _safe_get(step_data, "output", None) or step_data
        action_data = _safe_get(mo, "action", [None])[0] if _safe_get(mo, "action") else None
        
        # Get timestamp from metadata or use current time
        meta = _safe_get(step_data, "metadata", {})
        timestamp = _safe_get(meta, "step_start_time", None)
        if timestamp:
            try:
                ts = float(timestamp)
                utc_time = datetime.fromtimestamp(ts, timezone.utc)
                local_time = datetime.fromtimestamp(ts).astimezone()
            except (ValueError, TypeError):
                utc_time = datetime.now(timezone.utc)
                local_time = datetime.now().astimezone()
        else:
            utc_time = datetime.now(timezone.utc)
            local_time = datetime.now().astimezone()

        # Extract all action-related data
        def extract_action_data() -> tuple[str, Dict[str, Any]]:
            # Check all possible locations for action data
            action_data = None
            for source in [step_data, _safe_get(step_data, "output", {}), _safe_get(step_data, "model_output", {})]:
                if not isinstance(source, dict):
                    continue
                    
                # Look for action list
                if actions := _safe_get(source, "action", []):
                    if isinstance(actions, list) and actions:
                        action_data = actions[0]
                        break
                        
                # Look for direct action data
                if action_name := _safe_get(source, "action_name"):
                    action_data = {"action_name": action_name}
                    if args := _safe_get(source, "args"):
                        action_data["args"] = args
                    break
            
            action_data = action_data or {}
            
            # Gather all possible arguments
            all_args = {}
            # From action_data.args
            if isinstance(action_data, dict):
                all_args.update(_safe_get(action_data, "args", {}))
                # Direct properties that might be args
                for key in ["url", "text", "label", "value", "element"]:
                    if val := _safe_get(action_data, key):
                        all_args[key] = val
                        
            # Determine action type
            action = (_safe_get(action_data, "action_name", "") or "").lower()
            
            # If no action found, try to determine from args
            if not action or action == "unknown":
                if "url" in all_args:
                    action = "navigate_to"
                elif "text" in all_args and ("label" in all_args or "input" in str(all_args)):
                    action = "type_text"
                elif any(key in str(action_data).lower() for key in ["click", "button", "submit"]):
                    action = "click"
                elif "select" in str(action_data).lower():
                    action = "select"
                
            # Enhance args with element type if possible
            if "button" in str(all_args).lower() or "submit" in str(all_args).lower():
                all_args["element_type"] = "button"
            elif "link" in str(all_args).lower():
                all_args["element_type"] = "link"
            elif "input" in str(all_args).lower() or "field" in str(all_args).lower():
                all_args["element_type"] = "input"
                
            return action, all_args
            
        # Get enhanced action details
        action_name, args = extract_action_data()
        
        # Create human-readable description
        description = _format_human_description(action_name, args)
        
        # Merge all possible argument sources for maximum context
        args = {}
        args.update(_safe_get(action_data, "args", {}))
        args.update(_safe_get(action_data, "parameters", {}))
        for key in ["label", "text", "value", "url"]:
            if val := _safe_get(action_data, key):
                args[key] = val
                
        # Create action description
        action_desc = self._format_action_description(action_name, args)
        
        entry = {
            "timestamp_utc": utc_time.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "timestamp_local": local_time.strftime("%d-%m-%Y. %H.%M.%S"),
            "action_name": action_name,
            "description": action_desc,
            "details": args,
            "status": "Success" if success else "Failure",
            "error": error_msg if error_msg else None,
            "metadata": meta
        }
        self.actions.append(entry)

    def get_log(self, format: str = "json") -> str:
        """Get the action log in the specified format.
        
        Args:
            format: Output format - "json" or "human" (default: "json")
            
        Returns:
            Formatted log as either JSON or human-readable text
        """
        if format == "human":
            lines = []
            for entry in self.actions:
                # Use the pre-formatted description
                msg = f"{entry['timestamp_local']} - {entry['description']}"
                msg += f" ({entry['status']})"
                if entry['error']:
                    msg += f"\nError: {entry['error']}"
                lines.append(msg)
            return "\n".join(lines)
            
        return json.dumps(self.actions, indent=2, ensure_ascii=False)
def generate_action_log(history: Any, format: str = "json", include_metadata: bool = False) -> str:
    """Generate a clear, detailed log of test actions.

    Args:
        history: History object containing steps and any errors
        format: Output format - "json" or "human" (default: "json")
        include_metadata: Whether to include extra metadata in JSON output (default: False)

    Returns:
        Action log in the specified format (JSON string or human-readable text)
    """
    logger = ActionLogger()
    errors = _get_errors(history)
    steps = _get_steps(history)
    
    # Stop at first error if present
    error_index = -1
    for idx, error in enumerate(errors or []):
        if error:
            error_index = idx
            break

    # Process steps up to first error or all steps if no error
    for idx, step in enumerate(steps):
        if error_index != -1 and idx > error_index:
            break  # Stop at first error
        
        failed = bool(errors and idx < len(errors) and errors[idx])
        error_msg = str(errors[idx]) if failed else ""
        
        logger.log_action(
            step_data=step,
            success=not failed,
            error_msg=error_msg
        )

    log = logger.get_log(format)
    
    # Remove metadata if not requested
    if format == "json" and not include_metadata:
        entries = json.loads(log)
        for entry in entries:
            entry.pop("metadata", None)
        log = json.dumps(entries, indent=2, ensure_ascii=False)
    
    return log
def generate_human_readable_action_log(history: Any) -> str:
    """Generate a human-readable action log from test execution history.

    Args:
        history: History object containing steps and any errors

    Returns:
        Multi-line string with human-readable action entries and timestamps
    """
    # The human-readable format is now handled by ActionLogger's get_log method
    return generate_action_log(history, format="human")
def _format_human_description(action: str, args: Dict[str, Any]) -> str:
    """Format action and args into a human-readable description."""
    # Handle each action type specifically
    action = action.lower()
    
    if action == "navigate_to":
        return f"Navigated to {args.get('url', 'unknown URL')}"
        
    elif action == "type_text":
        text = args.get('text', '')
        field = args.get('label') or args.get('aria_label') or args.get('placeholder', '')
        if text and field:
            return f"Entered '{text}' in the '{field}' field"
        elif text:
            return f"Entered '{text}'"
        return "Entered text"
        
    elif action == "click":
        element = args.get('label') or args.get('text') or args.get('button') or args.get('link', '')
        element_type = args.get('element_type', '').lower()
        
        if element_type == 'button' and element:
            return f"Clicked the '{element}' button"
        elif element_type == 'link' and element:
            return f"Clicked the '{element}' link"
        elif element:
            return f"Clicked on '{element}'"
        return "Clicked element"
        
    elif action == "select":
        value = args.get('value', '')
        field = args.get('label', '')
        if value and field:
            return f"Selected '{value}' from '{field}'"
        elif value:
            return f"Selected '{value}'"
        return "Made selection"
        
    elif action == "wait":
        text = args.get('text', '')
        if text:
            return f"Waited for '{text}' to appear"
        return "Waited for element"
        
    elif action == "close":
        what = args.get('what', 'browser')
        return f"Closed {what}"
        
    # If we can't determine a specific format, but we have an action name
    elif action:
        return f"Performed {action}"
        
    # Ultimate fallback
    return "Performed browser action"

def _extract_action_details(step_data: Any) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Extract detailed action information from step data."""
    # Get all possible sources of action data
    model_output = _safe_get(step_data, "model_output", {}) or {}
    output = _safe_get(step_data, "output", {}) or {}
    
    # Extract action information from all possible locations
    action_info = None
    action_sources = [
        _safe_get(model_output, "action", []),
        _safe_get(output, "action", []),
        _safe_get(step_data, "action", []),
        [_safe_get(model_output, "command", {})],
        [_safe_get(output, "command", {})]
    ]
    
    for source in action_sources:
        if isinstance(source, list) and source:
            action_info = source[0]
            break
    
    if not action_info:
        action_info = {}
    
    # Get action name from multiple possible sources
    action_name = (
        _safe_get(action_info, "action_name") or
        _safe_get(action_info, "name") or
        _safe_get(action_info, "command") or
        _safe_get(action_info, "type") or
        _safe_get(step_data, "action_type") or
        ""
    ).lower()

    # Extract all possible arguments
    args = {}
    
    # From action_info args/parameters
    if isinstance(action_info, dict):
        args.update(_safe_get(action_info, "args", {}))
        args.update(_safe_get(action_info, "parameters", {}))
        
        # Get direct properties that might be arguments
        for key in ["url", "text", "value", "selector", "label", "element", 
                   "button", "link", "input", "field", "xpath"]:
            if val := _safe_get(action_info, key):
                args[key] = val

    # Try to infer action type if not explicitly specified
    if not action_name or action_name == "unknown":
        if "url" in args:
            action_name = "navigate"
        elif "text" in args and ("input" in str(args) or "field" in str(args)):
            action_name = "type"
        elif any(key in str(args).lower() for key in ["click", "button", "submit"]):
            action_name = "click"
        elif "select" in str(args).lower():
            action_name = "select"
            
    # Get metadata
    metadata = _safe_get(step_data, "metadata", {})
    
    return action_name, args, metadata

def generate_testing_report(task: str, history: Any) -> str:
    """Generate a detailed testing report with step-by-step analysis.

    Args:
        task: The test task description/objective
        history: History object containing test steps and errors

    Returns:
        String containing formatted test report with enhanced sections
    """
    # Create logger to get formatted action descriptions
    logger = ActionLogger()
    
    # 1. Test Overview Section
    overview = [
        "TEST EXECUTION REPORT",
        "=" * 50,
        "",
        "1. TEST OVERVIEW",
        "-" * 20,
        f"Task: {task}",
    ]
    
    # Get first action timestamp if available
    first_step = next(iter(_get_steps(history) or []), None)
    if first_step:
        meta = _safe_get(first_step, "metadata", {})
        ts = _safe_get(meta, "step_start_time", None)
        if ts:
            try:
                start_time = datetime.fromtimestamp(float(ts)).astimezone()
                overview.append(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            except (ValueError, TypeError):
                pass
    overview.append("")

    # 2. Test Steps Analysis
    steps = _get_steps(history)
    errors = _get_errors(history)
    total_steps = len([s for s in steps if _safe_get(_safe_get(s, "model_output", None) or 
                                                    _safe_get(s, "output", None) or s, "action", None)])
    
    # Find first failure
    failed_step_idx = None
    for idx, e in enumerate(errors):
        if e:
            failed_step_idx = idx
            break

    # 3. Test Summary Section
    summary = [
        "2. TEST SUMMARY",
        "-" * 20,
        f"Total Steps: {total_steps}",
        f"Steps Completed: {failed_step_idx if failed_step_idx is not None else total_steps}",
        f"Test Status: {'FAILED' if failed_step_idx is not None else 'PASSED'}",
        ""
    ]

    # 4. Test Steps Section
    steps_section = [
        "3. DETAILED TEST STEPS",
        "-" * 20
    ]

    for i, step in enumerate(_get_steps(history)):
        # Log the action to get its formatted description
        is_failed = bool(errors and i < len(errors) and errors[i])
        error_msg = str(errors[i]) if is_failed else ""
        logger.log_action(step, not is_failed, error_msg)
        
        # Get the latest action and ensure it has a proper description
        action_entry = logger.actions[-1]
        timestamp = action_entry['timestamp_local']
        description = action_entry['description']
        if description == "Performed":
            # Re-extract action details if description is too generic
            action_name, args, _ = _extract_action_details(step)
            description = logger._format_action_description(action_name, args)
        
        # Format step details with timestamp
        step_desc = f"Step {i+1}: [{timestamp}] {description}"
        
        # Add step status
        status = "❌ FAILED" if is_failed else "✓ PASSED"
        steps_section.append(f"{step_desc} [{status}]")
        
        # Add error details if failed
        if is_failed:
            error_msg = str(errors[i]).replace("\n", "\n      ")
            steps_section.append(f"      Error: {error_msg}")
        
        steps_section.append("")

    # 5. Failure Analysis Section (if applicable)
    failure_analysis = []
    if failed_step_idx is not None:
        failure_analysis = [
            "4. FAILURE ANALYSIS",
            "-" * 20
        ]
        
        # Use the formatted action entry from the logger
        failed_action = next((a for a in logger.actions if a.get('status') == 'Failure'), None)
        if failed_action:
            failure_analysis.extend([
                "Failed Action Details:",
                f"- Time: {failed_action['timestamp_local']}",
                f"- Action: {failed_action['action_name']}",
                f"- Description: {failed_action['description']}",
                f"- Details: {json.dumps(failed_action['details'], indent=2)}",
                f"- Error: {failed_action['error']}",
                "",
                "Test Execution Status:",
                f"- Failed at Step: {failed_step_idx + 1}",
                f"- Progress: {((failed_step_idx + 1) / total_steps * 100):.1f}% of test completed",
                ""
            ])

    # 6. Recommendations Section
    recommendations = [
        "5. RECOMMENDATIONS",
        "-" * 20
    ]

    if failed_step_idx is not None:
        error_msg = str(errors[failed_step_idx]).lower()
        if "timeout" in error_msg:
            recommendations.append("- Consider increasing the element wait timeout")
            recommendations.append("- Verify that the application's response time is within expected ranges")
        elif "element not found" in error_msg or "no element matches selector" in error_msg:
            recommendations.append("- Review and update element selectors")
            recommendations.append("- Verify that dynamic elements are fully loaded before interaction")
        elif "navigation" in error_msg:
            recommendations.append("- Add explicit waits after navigation events")
            recommendations.append("- Verify that page loads complete before proceeding")
        else:
            recommendations.append("- Review the error message and application logs")
            recommendations.append("- Consider adding additional error handling")
    else:
        recommendations.append("- Test passed successfully - no recommendations needed")

    # Combine all sections
    all_sections = overview + summary + steps_section
    if failure_analysis:
        all_sections.extend(failure_analysis)
    all_sections.extend(recommendations)

    return "\n".join(all_sections)
