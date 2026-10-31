# Reference prompts and private customization

The repository includes a small, newly written English reference pack in
[`forecasting/reference_prompts`](../forecasting/reference_prompts). It
illustrates the role interfaces and machine-readable response contracts. It is
not the operational prompt set used for the paper's experiments.

Design the instructions for your own task, input semantics, and model behavior.
The public templates contain no benchmark identifiers, example observations,
business thresholds, or task-specific operating instructions. Runtime context
is supplied separately by the caller.

## Files and roles

A prompt pack contains exactly these eight UTF-8 `.txt` files. Additional
`.txt` files are rejected:

| Files | Purpose | Response contract |
| --- | --- | --- |
| `forecast_system.txt`, `forecast_user.txt` | Numerical forecasting | One final `<answer>` element containing a JSON object with the supplied forecast keys and finite numeric values |
| `planning_system.txt`, `planning_user.txt` | Select an ordered tool plan | A JSON object whose `tools` property is an array of permitted tool names |
| `reflection_system.txt`, `reflection_user.txt` | Review a forecast against supplied requirements and evidence | A response beginning with `PASS` or `FAIL`, followed by a short reason |
| `exploration_system.txt`, `exploration_user.txt` | Propose alternative tool plans | A JSON object whose `strategies` property is an array of tool-name arrays |

Each user template must contain `{context}` exactly once. System templates do
not contain substitution fields. The caller inserts serialized JSON into
`{context}`, including the request information and applicable output contract.
Keep the placeholder intact and do not add other template variables. The exact
context fields depend on the calling role; they are not sample observations or
instructions to copy between requests.
Write literal template braces as `{{` and `}}`; JSON inserted through
`{context}` does not need this escaping.

The short English headers are reference notes and remain part of the template
text. The loader does not strip lines beginning with `#`, which could otherwise
remove an intentional Markdown instruction from a custom pack.

## Configure a private pack

Copy the reference files into a directory outside the source checkout:

```bash
mkdir -p ../castflow_private/prompts
cp forecasting/reference_prompts/*.txt ../castflow_private/prompts/
```

Edit the copies for the task. Keep each role's response contract compatible with
the parser and preserve the user-template placeholder. Select the directory
through the environment:

```bash
export CASTFLOW_PROMPT_DIR="$(cd ../castflow_private/prompts && pwd)"
```

Alternatively, append `--prompt-dir /absolute/path/to/private/prompts` to the
workflow command. The explicit CLI directory overrides the environment setting.
With neither setting, CastFlow loads the packaged reference templates. A custom
pack must be complete; missing or malformed files are rejected rather than
silently mixed with another pack.

Prompt text is distinct from structured request data. Keep observations, tool
results, retrieved records, and feedback in the runtime context. Use the private
templates for task-specific instructions. Do not replace the current request's
context with archived examples or unknown future observations.

## Storage and runtime identity

Prompt identities record the selected mode and content fingerprints. They do
not embed the prompt bodies in the identity itself. Editing a template changes
that identity, so recovery cannot silently reuse a response from a different
prompt pack.

Rendered prompts can still be present in generated memory, training examples,
and execution journals because those artifacts support the workflow. Keep the
custom templates and generated runtime artifacts outside the public checkout.
Publish the generic reference pack separately from task-specific instructions,
data, credentials, or captured model conversations.
