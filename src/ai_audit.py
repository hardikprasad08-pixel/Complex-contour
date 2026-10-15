import json
import os
from datetime import datetime
from typing import Dict, Any, List

class AIAuditTrail:
    """
    Tracks, logs, and maintains a structured JSON-based audit trail tracking 
    user inputs, parsed expressions, decision engine metrics, and calculation steps.
    """
    def __init__(self, log_filepath: str = 'data/audit_trail.json'):
        self.log_filepath = log_filepath
        os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)
        if not os.path.exists(self.log_filepath):
            with open(self.log_filepath, 'w') as f:
                json.dump([], f)

    def log_event(self, expression: str, contour_info: Dict[str, Any], 
                  decision_info: Dict[str, Any], results_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Logs a session run event to the structured audit trail file.
        """
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'expression_str': expression,
            'contour': contour_info,
            'decision_engine': decision_info,
            'results': results_info
        }

        try:
            with open(self.log_filepath, 'r') as f:
                logs = json.load(f)
        except Exception:
            logs = []

        logs.append(event)

        with open(self.log_filepath, 'w') as f:
            json.dump(logs, f, indent=4)
            
        return event

    def get_logs(self) -> List[Dict[str, Any]]:
        """
        Retrieves the complete audit trail history.
        """
        if not os.path.exists(self.log_filepath):
            return []
        try:
            with open(self.log_filepath, 'r') as f:
                return json.load(f)
        except Exception:
            return []
