# Environment Configuration

Environment configurations define tools, resources, and prompts that are available during agent execution. Unlike Knowledge sources (which are loaded at startup), environment resources and prompts are loaded on-demand when tools request them.

## Example Configuration

```yaml
agents:
  code_analyzer:
    environment:
      type: "inline"  # or "file" to load from separate file

      # Tool Definitions
      tools:
        analyze_complexity:
          import_path: "myapp.tools:analyze_complexity"
          description: "Analyze code complexity"

        git_blame:
          import_path: "myapp.git:blame"
          description: "Get git blame information"
          requires_confirmation: true

        format_code:
          import_path: "black.format_str"
          description: "Format Python code using black"
          settings:
            line_length: 88

      # On-demand Resources
      resources:
        python_docs:
          type: "path"
          path: "docs/python/*.md"
          watch: true  # Update when files change

        code_standards:
          type: "text"
          content: "Our coding standards..."

        test_results:
          type: "cli"
          command: "pytest --json"
          refresh_interval: 300  # seconds

        git_info:
          type: "callable"
          import_path: "myapp.git:get_info"

      # On-demand Prompts
      prompts:
        bug_analysis:
          type: "file"
          path: "prompts/bug_analysis.txt"

        code_review:
          type: "dynamic"
          import_path: "myapp.prompts:generate_review_prompt"
```

## File-based Environment

You can also move the environment configuration to a separate file:

```yaml
agents:
  code_analyzer:
    environment:
      type: "file"
      uri: "environments/code_analyzer.yml"
```

## Components

### Tools
- Python functions that agents can call
- Can be imported from any installed package
- Support settings and confirmation requirements
- Can be chained together by capable agents

### Resources (On-demand)
Resources that are loaded only when requested by tools:
- `path`: File patterns with optional watching
- `text`: Inline text content
- `cli`: Command output with refresh intervals
- `callable`: Python function results
- `repository`: Git repositories
- `source`: Python source code

### Prompts (On-demand)
Prompts that are rendered when needed:
- `file`: Load from text files
- `dynamic`: Generate using Python functions
- `template`: Jinja2 templates with context
- `static`: Inline prompt definitions

## Difference to Knowledge Sources

- **Environment**: On-demand loading when tools need them
  ```yaml
  environment:
    resources:
      docs:
        type: "path"
        path: "*.md"  # Loaded when a tool requests it
  ```

- **Knowledge**: Pre-loaded at agent startup
  ```yaml
  knowledge:
    paths: ["docs/**/*.md"]  # Loaded when agent starts
  ```

## Example With Separate File

`environments/code_analyzer.yml`:
```yaml
tools:
  analyze_complexity:
    import_path: "myapp.tools:analyze_complexity"

resources:
  docs:
    type: "path"
    path: "*.md"

prompts:
  code_review:
    type: "file"
    path: "prompts/review.txt"
```
