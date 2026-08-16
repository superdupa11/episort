---
description: Capture a bug, feature idea, or general idea into Tududi as a task under the EpiSort project. Use whenever you notice a bug that isn't being fixed right now, a feature idea comes up in conversation, or a stray idea worth remembering surfaces — in this repo or while discussing it. Don't use for something you're about to fix or implement immediately in this session.
---

You are doing a quick, low-friction capture into Tududi — not a context switch. This should take one tool call and a one-line confirmation, then work resumes immediately.

## Steps

1. Pick the closest type: `bug` (something broken), `feature` (something new/improved worth building), or `idea` (a thought, question, or possibility not yet scoped as either). Default to `idea` if genuinely unsure.

2. Write a self-contained summary — someone reading it weeks from now, with no memory of this session, needs to understand it. Include a file/line reference for bugs when you have one (`path/to/file.py:123`), and a one-sentence "why" for features/ideas.

3. Call `mcp__tududi__create_task`:
   - `name`: a short actionable title for the summary (~80 chars)
   - `description`: `[<type>] repo: episort — <full summary>` (keep this exact prefix so the item stays traceable back to its source repo/type even after filing)
   - `project_id`: 10 (EpiSort)
   - `tags`: `[<type>]` (i.e. `["bug"]`, `["feature"]`, or `["idea"]`)
   - `priority`: leave unset by default; set `"high"` only if the summary itself signals urgency (data loss, security, crash, prod-down) — don't infer urgency from tone alone

4. Tell the user one line confirming what was captured (task name + project), then continue whatever you were doing before. Do not open Tududi, do not triage it now, do not let this become the main thread of the conversation.

## What NOT to do

- Don't use this for something you're fixing right now in this session — just fix it.
- Don't batch multiple unrelated items into one task — one capture per bug/feature/idea.
- Don't check for existing near-duplicate tasks first — this is a fast one-shot capture, not a triage pass. Merge duplicates by hand later if any turn up.
- Don't second-guess the project mapping above — project 10 (EpiSort) is fixed for this repo.

`mcp__tududi__add_to_inbox` and the `tududi-triage` skill still exist for anything captured by hand outside this flow (e.g. via Tududi's own mobile/web client) — this skill no longer routes through the inbox.
