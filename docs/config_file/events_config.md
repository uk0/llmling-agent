# Event Configuration

Events (or "triggers") define automated activations of agents based on various sources. They allow agents to respond to:
- File system changes
- Webhook calls
- Email notifications
- Manual triggers
- (Future: Message queues, time schedules, etc.)


## Common Properties
All event sources share these base properties:

```yaml
triggers:
  - name: "my_trigger"                   # Unique identifier
    enabled: true                        # Whether trigger is active
    template: <a_jinja_template_string>  # Optional event formatting
    include_metadata: true               # Include event metadata
    include_timestamp: true              # Include event timestamp
```

## File Watch Events
Monitor file system changes:

```yaml
triggers:
  - type: "file"
    name: "python_watcher"
    paths: ["src/**/*.py"]           # Paths to watch
    extensions: [".py"]              # Optional file type filter
    ignore_paths: ["**/__pycache__"] # Optional ignore patterns
    recursive: true                  # Watch subdirectories
    debounce: 1600                  # Minimum ms between triggers
```

## Webhook Events
Listen for HTTP requests:

```yaml
triggers:
  - type: "webhook"
    name: "github_webhook"
    port: 8000
    path: "/github"
    secret: "${WEBHOOK_SECRET}"      # Optional validation secret
```

## Email Events
Monitor email inbox:

```yaml
triggers:
  - type: "email"
    name: "support_inbox"
    host: "imap.gmail.com"          # IMAP server hostname
    port: 993                       # Server port (993 for SSL)
    username: "support@domain.com"
    password: "${EMAIL_PASSWORD}"
    folder: "INBOX"                 # Mailbox to monitor
    ssl: true                       # Use SSL connection
    check_interval: 60              # Seconds between checks
    mark_seen: true                # Mark processed emails as seen
    filters:                       # Optional email filters
      from: "important@client.com"
      subject: "urgent"
    max_size: 1048576             # Max email size in bytes
```

## Time Events
Schedule regular agent actions:

```yaml
triggers:
  - type: "time"
    name: "daily_report"
    schedule: "0 9 * * 1-5"      # Cron expression: 9am on weekdays
    timezone: "UTC"              # Optional timezone (defaults to system)

  - type: "time"
    name: "quick_check"
    interval: "5m"               # Simple interval: 5m, 1h, 2d etc.
    jitter: "30s"               # Optional random delay

  - type: "time"
    name: "monitoring"
    schedule:
      - "0 9 * * 1-5"           # Multiple schedules
      - "0 18 * * 1-5"
    skip_missed: true           # Don't run missed executions
```


## Multiple Events
Agents can have multiple events of different types:

```yaml
agents:
  project_assistant:
    triggers:
      # Watch for code changes
      - type: "file"
        name: "code_watcher"
        paths: ["src/**/*.py"]

      # Monitor support inbox
      - type: "email"
        name: "support"
        host: "imap.gmail.com"
        username: "${EMAIL_USER}"
        password: "${EMAIL_PASS}"

      # Listen for webhooks
      - type: "webhook"
        name: "github_events"
        port: 8000
        path: "/github"
```

## Event Processing
1. Event source detects change and creates event
2. Event's `to_prompt()` generates core message
3. Template wraps message with optional metadata/timestamp
4. Formatted event is sent to agent for processing
