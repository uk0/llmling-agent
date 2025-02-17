"""SQL queries for Supabase operations."""

CREATE_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS messages (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id uuid NOT NULL,
    timestamp timestamptz NOT NULL DEFAULT now(),
    role text NOT NULL,
    name text,
    content text NOT NULL,
    model text,
    model_provider text,
    model_name text,
    forwarded_from text[],
    total_tokens integer,
    prompt_tokens integer,
    completion_tokens integer,
    cost decimal,
    response_time decimal,
    metadata jsonb
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp
ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_role
ON messages(role);
"""

CREATE_CONVERSATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS conversations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name text NOT NULL,
    start_time timestamptz NOT NULL DEFAULT now(),
    total_tokens integer DEFAULT 0,
    total_cost decimal DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_conversations_agent
ON conversations(agent_name);
CREATE INDEX IF NOT EXISTS idx_conversations_start_time
ON conversations(start_time);
"""

CREATE_TOOL_CALLS_TABLE = """
CREATE TABLE IF NOT EXISTS tool_calls (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id uuid NOT NULL,
    message_id uuid NOT NULL,
    tool_call_id text,
    timestamp timestamptz NOT NULL DEFAULT now(),
    tool_name text NOT NULL,
    args jsonb NOT NULL,
    result text NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_conversation
ON tool_calls(conversation_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_message
ON tool_calls(message_id);
"""

CREATE_COMMANDS_TABLE = """
CREATE TABLE IF NOT EXISTS commands (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id text NOT NULL,
    agent_name text NOT NULL,
    command text NOT NULL,
    timestamp timestamptz NOT NULL DEFAULT now(),
    context_type text,
    metadata jsonb
);

CREATE INDEX IF NOT EXISTS idx_commands_session
ON commands(session_id);
CREATE INDEX IF NOT EXISTS idx_commands_agent
ON commands(agent_name);
"""
