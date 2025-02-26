from __future__ import annotations

import functools
import os
from typing import Literal

from llmling_agent_tools.jira_tool.formatter import format_issue


JIRA_EMAIL = "philipptemminghoff@gmail.com"
JIRA_TOKEN = os.getenv("JIRA_API_KEY")
SERVER = "https://philipptemminghoff.atlassian.net"
PROJECT = "SCRUM"
assert JIRA_TOKEN


@functools.cache
def get_client():
    import jira

    return jira.JIRA(server=SERVER, basic_auth=(JIRA_EMAIL, JIRA_TOKEN))  # pyright: ignore


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
