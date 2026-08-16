---
description: Process Tududi's inbox into properly typed, tagged tasks under the right project. Use when asked to triage Tududi, clean up the inbox, review captured items, or periodically (e.g. weekly) to convert captured bugs/features/ideas into actionable tasks.
---

You are triaging Tududi's inbox. `tududi-capture` now files tasks directly into the right project instead of writing here, so items landing in the inbox going forward are almost always entered by hand — via Tududi's own mobile/web client, or added some other way outside the capture flow. The inbox is still global to the Tududi instance, not scoped to whichever repo you're running in, so a triage run here may surface items related to any of Brian's active repos. Process all of it, not just items related to the current repo.

## Repo → project mapping

| repo slug | Tududi project | project_id |
|---|---|---|
| manabase-central | Manabase Central | 8 |
| manabase-lgs | Manabase LGS | 2 |
| manabase-upkeep | Manabase Upkeep | 1 |
| movienight | binger | 9 |
| episort | EpiSort | 10 |

If a repo gets renamed or a new repo starts using `tududi-capture`, this table (and the copy of it in every repo's `tududi-triage.md`) needs a matching row — update all copies together.

## Steps

1. `mcp__tududi__list_inbox` (paginate with `offset` if `count` equals the page `limit`) to pull every item.

2. For each item, check whether `content` matches the legacy pattern `[<type>] repo: <slug> — <summary>` that `tududi-capture` used to write before it switched to filing tasks directly:
   - `type` → bug / feature / idea
   - `slug` → look up in the table above for the target project
   - `summary` → becomes the task

   If an item **doesn't** match this pattern (e.g. captured by hand outside the skill, or from Tududi's own mobile/web client), don't guess — set it aside and ask the user how to categorize it rather than silently filing it somewhere wrong.

3. Before creating a task, check for a near-duplicate already open in that project (`mcp__tududi__list_tasks` with that `project_id`, status not done/cancelled). If one clearly covers the same ground, skip creating a new task and just `process_inbox_item` the inbox entry, noting the existing task in your report instead.

4. Otherwise, `mcp__tududi__create_task`:
   - `name`: the summary, trimmed to a short actionable title (~80 chars; move any overflow detail into `description`)
   - `description`: the full original inbox content, so nothing gets lost in the trim
   - `project_id`: from the mapping table
   - `tags`: `[type]` (i.e. `["bug"]`, `["feature"]`, or `["idea"]`)
   - `priority`: leave unset by default; set `"high"` only if the summary itself signals urgency (data loss, security, crash, prod-down) — don't infer urgency from tone alone

5. `mcp__tududi__process_inbox_item` on the source inbox item once its task exists (or once you've confirmed it's a duplicate per step 3).

## After triaging

Report a short summary: how many items processed, broken down by project and type, any duplicates skipped (with which existing task), and anything set aside for the user to categorize by hand. Don't leave ambiguous items processed-and-guessed — leave them in the inbox (untouched status) until the user weighs in.
