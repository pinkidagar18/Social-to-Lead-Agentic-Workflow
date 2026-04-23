"""
Tool: mock_lead_capture
Simulates a CRM lead capture API call.
Only called after name, email, and platform are all confirmed.
"""

import json
import re
from datetime import datetime
from pathlib import Path


def validate_email(email: str) -> bool:
    """Basic email format validation."""
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email.strip()))


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Mock CRM API call that captures a qualified lead.
    Prints confirmation and saves to a local leads log file.

    Args:
        name:     Full name of the lead
        email:    Email address of the lead
        platform: Creator platform (YouTube, Instagram, TikTok, etc.)

    Returns:
        dict with status and lead_id
    """
    # Validate inputs
    if not name or not name.strip():
        return {"status": "error", "message": "Name cannot be empty."}

    if not validate_email(email):
        return {"status": "error", "message": f"Invalid email format: {email}"}

    if not platform or not platform.strip():
        return {"status": "error", "message": "Platform cannot be empty."}

    # Simulate lead ID generation
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    lead_id = f"LEAD-{timestamp}-{name[:3].upper()}"

    lead_data = {
        "lead_id": lead_id,
        "name": name.strip(),
        "email": email.strip().lower(),
        "platform": platform.strip(),
        "captured_at": datetime.now().isoformat(),
        "source": "autostream_agent",
        "status": "qualified",
    }

    # Console output (as required by assignment)
    print("\n" + "=" * 50)
    print("  LEAD CAPTURED SUCCESSFULLY")
    print("=" * 50)
    print(f"  Lead ID  : {lead_id}")
    print(f"  Name     : {lead_data['name']}")
    print(f"  Email    : {lead_data['email']}")
    print(f"  Platform : {lead_data['platform']}")
    print(f"  Time     : {lead_data['captured_at']}")
    print("=" * 50 + "\n")

    # Persist to local leads log
    log_path = Path("leads_log.json")
    existing_leads = []

    if log_path.exists():
        with open(log_path, "r") as f:
            try:
                existing_leads = json.load(f)
            except json.JSONDecodeError:
                existing_leads = []

    existing_leads.append(lead_data)

    with open(log_path, "w") as f:
        json.dump(existing_leads, f, indent=2)

    return {
        "status": "success",
        "lead_id": lead_id,
        "message": f"Lead captured successfully for {name}.",
    }
