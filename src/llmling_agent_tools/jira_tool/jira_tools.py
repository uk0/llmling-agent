from __future__ import annotations

from datetime import datetime
import functools
import os
from typing import Any, Literal


JIRA_EMAIL = "philipptemminghoff@gmail.com"
JIRA_TOKEN = os.getenv("JIRA_API_KEY")
SERVER = "https://philipptemminghoff.atlassian.net"
PROJECT = "SCRUM"
assert JIRA_TOKEN


@functools.cache
def get_client():
    import jira

    return jira.JIRA(server=SERVER, basic_auth=(JIRA_EMAIL, JIRA_TOKEN))  # pyright: ignore


def format_issue_field(field: Any) -> str:
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


def format_date(date_str: str | None) -> str:
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
            formatted_value = format_date(value)
        elif field_name == "labels":
            formatted_value = ", ".join(value) if value else ""
        else:
            formatted_value = format_issue_field(value)

        if formatted_value:
            result.append(f"{display_name}: {formatted_value}")

    # Process comments
    comment_obj = getattr(issue_field, "comment", None)
    comments = getattr(comment_obj, "comments", []) if comment_obj else []

    if comments:
        result.append("Comments:")
        for comment in comments:
            author = getattr(comment, "author", None)
            author_name = format_issue_field(author) if author else "Unknown"
            comment_date = format_date(getattr(comment, "created", ""))
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


def search_for_issues(jql_str: str) -> str:
    """Search for issues in Jira Ticket system.

    Args:
        jql_str: The JQL query string.

    Returns:
        str: A result for given query
    """
    from pydantic_ai import ModelRetry

    client = get_client()
    try:
        issues = client.search_issues(jql_str=jql_str)
    except Exception as e:  # noqa: BLE001
        raise ModelRetry(str(e)) from None

    if not issues:
        return "No issues found matching the query."

    formatted_issues = []
    for i in issues:
        issue = client.issue(i.key)  # type: ignore
        formatted_issues.append(f"Issue: {issue.key}\n{format_issue(issue.fields)}")

    return "\n\n---\n\n".join(formatted_issues)


def create_issue(
    summary: str,
    description: str,
    issuetype: Literal["Bug"] = "Bug",
    attachment: str | None = None,
):
    """Create a new issue in Jira.

    Args:
        summary: The issue summary.
        description: The issue description.
        issuetype: The issue type.
        attachment: Optional attachment for the ticket

    Returns:
        str: A message indicating the success of the operation.
    """
    issue_type = {"name": issuetype}
    client = get_client()
    issue = client.create_issue(
        project=PROJECT,
        summary=summary,
        description=description,
        issuetype=issue_type,
    )
    if attachment:
        client.add_attachment(issue.id, attachment, "attachment.txt")

    return f"Issue {issue.id} ({issue.key}) created successfully"


if __name__ == "__main__":

    async def main():
        from llmling_agent import Agent

        agent = Agent[None](model="gpt-4o-mini", tools=[search_for_issues])
        result = await agent.run(
            "Search for the ticket SCRUM-6 using JIRA query syntax."
            "Return all information completely untouched."
        )
        # result = await agent.run("Create a ticket in jira with a random description")
        print(result.format())
        # print(create_issue("SCRUM", "Something broken", "fdsfhkjdsfldsjfl"))

    import asyncio

    asyncio.run(main())
