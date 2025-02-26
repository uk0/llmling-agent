# JQL: Jira Query Language Guide

## Overview
Jira Query Language (JQL) is a powerful SQL-like syntax for searching Jira issues. It enables precise filtering through structured queries instead of simple keyword searches.

## Basic Structure
A JQL query follows this pattern:
```
field operator value [AND|OR field operator value...] [ORDER BY field [ASC|DESC]]
```

## Core Syntax Elements

### Fields
Standard fields include:
- `project`: Project identifier
- `assignee`: Person assigned to issue
- `reporter`: Person who created the issue
- `status`: Current status (e.g., "Open", "In Progress", "Done")
- `priority`: Issue priority level
- `text`: Searches across all text fields

### Operators
- `=`, `!=`: Equals, not equals
- `>`, `>=`, `<`, `<=`: Greater/less than comparisons (for dates, numbers)
- `IN`: Value in a specified list
- `~` (CONTAINS): Text search (supports wildcards)
- `IS NULL`, `IS NOT NULL`: Empty/not empty field check

### Boolean Operators
- `AND`: Both conditions must be true
- `OR`: Either condition can be true
- `NOT`: Negates a condition

### Values
- Text values require quotes: `summary ~ "login bug"`
- Multiple values use parentheses: `status IN ("Open", "In Progress")`
- Date formats: `created > "2023-01-01"`
- Special functions: `assignee = currentUser()`

## Examples

```
# All issues in the TEST project
project = "TEST"

# High priority bugs assigned to me
project = "TEST" AND type = Bug AND priority = High AND assignee = currentUser()

# Issues created in the last week with "login" in description
created >= -1w AND description ~ "login"

# All unresolved issues in dev team projects
project IN (Backend, Frontend) AND resolution = Unresolved

# Tasks ordered by priority then creation date
type = Task ORDER BY priority DESC, created ASC
```

## Text Searching Tips

- Exact phrase: `summary ~ "\"exact phrase\""`
- Wildcard: `summary ~ "log*"` (matches "login", "logout", etc.)
- Simple text: `text ~ "database error"`
- Excluded term: `text ~ "database" AND NOT text ~ "timeout"`

## Advanced Features

- Grouping with parentheses: `status = Resolved AND (priority = High OR assignee = currentUser())`
- Functions: `duedate < endOfWeek()`, `updatedDate > startOfMonth(-1)`
- Labels and custom fields: `labels = "frontend" AND cf[12345] = "value"`

Remember to escape special characters with backslashes and quote reserved words when using them as values.
