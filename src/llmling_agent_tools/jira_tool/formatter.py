from __future__ import annotations

from datetime import datetime
from typing import Any


def _format_issue_field(field: Any) -> str:
    """Format a single issue field into a readable string."""
    if field is None:
        return ""

    # Handle common Jira resource objects
    for attr in ("displayName", "name", "summary", "value"):
        if hasattr(field, attr):
            return getattr(field, attr)

    # Handle lists of resources
    if isinstance(field, list):
        if not field:
            return ""

        # Check first item for common attributes
        first_item = field[0]
        for attr in ("name", "displayName"):
            if hasattr(first_item, attr):
                return ", ".join(getattr(item, attr) for item in field)

        return ", ".join(str(item) for item in field)

    return str(field)


def _format_date(date_str: str | None) -> str:
    """Format a date string to a more readable format."""
    if not date_str:
        return ""
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f%z")
        return date_obj.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return date_str


def format_issue(issue_field: Any) -> str:
    """Format a Jira issue field object into a readable text format."""
    result = []

    # Get field mappings with friendly names
    field_mapping = {
        "summary": "Summary",
        "issuetype": "Type",
        "status": "Status",
        "priority": "Priority",
        "assignee": "Assignee",
        "reporter": "Reporter",
        "created": "Created",
        "duedate": "Due Date",
        "description": "Description",
        "labels": "Labels",
    }

    # Process simple fields
    for field_name, display_name in field_mapping.items():
        value = getattr(issue_field, field_name, None)
        if not value:
            continue

        if field_name in ("created", "duedate"):
            formatted_value = _format_date(value)
        elif field_name == "labels":
            formatted_value = ", ".join(value) if value else ""
        else:
            formatted_value = _format_issue_field(value)

        if formatted_value:
            result.append(f"{display_name}: {formatted_value}")

    # Process comments
    comment_obj = getattr(issue_field, "comment", None)
    comments = getattr(comment_obj, "comments", []) if comment_obj else []

    if comments:
        result.append("Comments:")
        for comment in comments:
            author = getattr(comment, "author", None)
            author_name = _format_issue_field(author) if author else "Unknown"
            comment_date = _format_date(getattr(comment, "created", ""))
            body = getattr(comment, "body", "")
            result.append(f"  - {author_name} ({comment_date}): {body}")

    # Process issue links
    links = getattr(issue_field, "issuelinks", [])
    if links:
        result.append("Issue Links:")
        for link in links:
            if hasattr(link, "outwardIssue"):
                link_type = getattr(link.type, "outward", "relates to")
                result.append(f"  - {link_type}: {link.outwardIssue.key}")
            elif hasattr(link, "inwardIssue"):
                link_type = getattr(link.type, "inward", "relates to")
                result.append(f"  - {link_type}: {link.inwardIssue.key}")

    return "\n".join(result)
