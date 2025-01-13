# Response Types

Response types define structured output formats for agents. They can be defined directly in YAML or imported from Python code.

!!! tip "Type Safety"
    While YAML configuration is convenient, defining response types as Pydantic models in Python code provides better type safety, IDE support, and reusability:
    ```python
    from pydantic import BaseModel

    class AnalysisResult(BaseModel):
        success: bool
        issues: list[str]
        severity: str
    ```

## Inline Responses
Define response structure directly in YAML:

### Simple Status Response
```yaml
responses:
  StatusResponse:
    type: "inline"
    description: "Simple operation result with status"
    fields:
      success:
        type: "bool"
        description: "Whether operation succeeded"
      message:
        type: "str"
        description: "Status message or error details"
```

### Analysis Result
```yaml
responses:
  CodeAnalysis:
    type: "inline"
    description: "Code analysis results with issues"
    fields:
      issues:
        type: "list[str]"
        description: "List of found issues"
      severity:
        type: "str"
        description: "Overall severity level"
      locations:
        type: "list[str]"
        description: "Source code locations"
```

### Complex Response
```yaml
responses:
  DataProcessingResult:
    type: "inline"
    description: "Complex data processing result"
    fields:
      success:
        type: "bool"
        description: "Operation success"
      records_processed:
        type: "int"
        description: "Number of processed records"
      errors:
        type: "list[str]"
        description: "List of errors if any"
      metrics:
        type: "dict[str, float]"
        description: "Processing metrics"
```

## Imported Responses
Import response types from Python code:

### Python Type Import
```yaml
responses:
  AdvancedAnalysis:
    type: "import"
    import_path: "myapp.types:AnalysisResult"
```

### Package Response Type
```yaml
responses:
  MetricsResult:
    type: "import"
    import_path: "myapp.analysis:MetricsResponse"
```

## Using Response Types

### Assign to Agent
```yaml
agents:
  analyzer:
    model: "openai:gpt-4"
    result_type: "CodeAnalysis"  # Reference response by name
```

### Inline with Custom Tool Name
```yaml
agents:
  processor:
    result_type:
      type: "inline"  # Direct inline definition
      result_tool_name: "create_result"  # Custom tool name
      result_tool_description: "Create the final analysis result"
      fields:
        success:
          type: "bool"
        details:
          type: "str"
```

## Available Field Types
- `str`: Text strings
- `int`: Integer numbers
- `float`: Floating point numbers
- `bool`: Boolean values
- `list[type]`: Lists of values (e.g., `list[str]`, `list[int]`)
- `dict[key_type, value_type]`: Dictionaries
- `datetime`: Date and time values
- Custom types through imports
