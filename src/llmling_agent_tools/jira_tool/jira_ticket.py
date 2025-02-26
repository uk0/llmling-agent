from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    import jira


class JiraUser(BaseModel):
    """Represents a Jira user."""

    display_name: str
    name: str | None = None
    email_address: str | None = None


class JiraStatus(BaseModel):
    """Represents a Jira status."""

    name: str
    description: str | None = None
    category_name: str | None = None


class JiraIssueType(BaseModel):
    """Represents a Jira issue type."""

    name: str
    description: str | None = None


class JiraPriority(BaseModel):
    """Represents a Jira priority."""

    name: str


class JiraProject(BaseModel):
    """Represents a Jira project."""

    key: str
    name: str


class JiraComment(BaseModel):
    """Represents a Jira comment."""

    body: str
    author: JiraUser
    created: datetime | None = None
    updated: datetime | None = None


class JiraIssueLink(BaseModel):
    """Represents a Jira issue link."""

    link_type: str
    issue_key: str
    direction: Literal["inward", "outward"]


class JiraTicket(BaseModel):
    """Represents a Jira ticket with all its fields."""

    key: str
    summary: str
    description: str | None = None
    issue_type: JiraIssueType
    status: JiraStatus
    priority: JiraPriority | None = None
    assignee: JiraUser | None = None
    reporter: JiraUser | None = None
    project: JiraProject
    created: datetime | None = None
    updated: datetime | None = None
    due_date: datetime | None = None
    labels: list[str] = Field(default_factory=list)
    comments: list[JiraComment] = Field(default_factory=list)
    issue_links: list[JiraIssueLink] = Field(default_factory=list)

    def format(self) -> str:
        """Format the ticket as a readable string."""
        lines = [
            f"Issue: {self.key}",
            f"Summary: {self.summary}",
            f"Type: {self.issue_type.name}",
            f"Status: {self.status.name}",
        ]

        if self.priority:
            lines.append(f"Priority: {self.priority.name}")

        if self.assignee:
            lines.append(f"Assignee: {self.assignee.display_name}")

        if self.reporter:
            lines.append(f"Reporter: {self.reporter.display_name}")

        if self.created:
            lines.append(f"Created: {self.created.strftime('%Y-%m-%d %H:%M')}")

        if self.due_date:
            lines.append(f"Due Date: {self.due_date.strftime('%Y-%m-%d')}")

        if self.description:
            lines.append(f"Description: {self.description}")

        if self.labels:
            lines.append(f"Labels: {', '.join(self.labels)}")

        if self.comments:
            lines.append("Comments:")
            for comment in self.comments:
                author_name = comment.author.display_name
                created_str = (
                    comment.created.strftime("%Y-%m-%d %H:%M")
                    if comment.created
                    else "Unknown date"
                )
                lines.append(f"  - {author_name} ({created_str}): {comment.body}")

        if self.issue_links:
            lines.append("Issue Links:")
            for link in self.issue_links:
                lines.append(f"  - {link.link_type}: {link.issue_key}")  # noqa: PERF401

        return "\n".join(lines)


def parse_jira_date(date_str: str | None) -> datetime | None:
    """Parse a Jira date string into a datetime object."""
    if not date_str:
        return None

    try:
        # Handle standard Jira datetime format with timezone
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f%z")
    except ValueError:
        try:
            # Handle date-only format
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return None


def extract_jira_ticket(issue: jira.Issue) -> JiraTicket:
    """Extract a structured JiraTicket from a Jira issue object."""
    fields = issue.fields

    # Extract user information
    assignee = None
    if fields.assignee:
        assignee = JiraUser(
            display_name=fields.assignee.displayName,
            name=fields.assignee.name,
            email_address=getattr(fields.assignee, "emailAddress", None),
        )

    reporter = None
    if fields.reporter:
        reporter = JiraUser(
            display_name=fields.reporter.displayName,
            name=fields.reporter.name,
            email_address=getattr(fields.reporter, "emailAddress", None),
        )

    # Extract status information
    status = JiraStatus(
        name=fields.status.name, description=getattr(fields.status, "description", None)
    )

    if hasattr(fields.status, "statusCategory"):
        status.category_name = fields.status.statusCategory.name

    # Extract issue type
    issue_type = JiraIssueType(
        name=fields.issuetype.name,
        description=getattr(fields.issuetype, "description", None),
    )

    # Extract priority
    priority = None
    if hasattr(fields, "priority") and fields.priority:
        priority = JiraPriority(name=fields.priority.name)

    # Extract project
    project = JiraProject(key=fields.project.key, name=fields.project.name)

    # Extract comments
    comments = []
    if hasattr(fields, "comment") and hasattr(fields.comment, "comments"):
        for comment in fields.comment.comments:
            comment_author = JiraUser(
                display_name=comment.author.displayName,
                name=comment.author.name,
                email_address=getattr(comment.author, "emailAddress", None),
            )
            comment = JiraComment(
                body=comment.body,
                author=comment_author,
                created=parse_jira_date(getattr(comment, "created", None)),
                updated=parse_jira_date(getattr(comment, "updated", None)),
            )
            comments.append(comment)

    # Extract issue links
    issue_links = []
    if hasattr(fields, "issuelinks"):
        for link in fields.issuelinks:
            if hasattr(link, "outwardIssue"):
                link = JiraIssueLink(
                    link_type=link.type.outward,
                    issue_key=link.outwardIssue.key,
                    direction="outward",
                )
                issue_links.append(link)
            elif hasattr(link, "inwardIssue"):
                link = JiraIssueLink(
                    link_type=link.type.inward,
                    issue_key=link.inwardIssue.key,
                    direction="inward",
                )
                issue_links.append(link)

    # Create the ticket
    return JiraTicket(
        key=issue.key,
        summary=fields.summary,
        description=getattr(fields, "description", None),
        issue_type=issue_type,
        status=status,
        priority=priority,
        assignee=assignee,
        reporter=reporter,
        project=project,
        created=parse_jira_date(getattr(fields, "created", None)),
        updated=parse_jira_date(getattr(fields, "updated", None)),
        due_date=parse_jira_date(getattr(fields, "duedate", None)),
        labels=getattr(fields, "labels", []),
        comments=comments,
        issue_links=issue_links,
    )
