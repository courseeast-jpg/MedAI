# Codex Detached Worktree Autopush Rule

Codex detached worktrees are acceptable for MedAI implementation blocks.
An empty `git branch --show-current` result is acceptable when the HEAD,
remote, ancestry, and working-tree checks pass.

For future MedAI implementation blocks, the final commit may be pushed
automatically from a detached Codex worktree after all of these conditions are
true:

1. Required focused tests pass.
2. The block validation script passes.
3. The public report privacy check passes.
4. `external_api_used=false`.
5. `auto_accept=false`.
6. Before commit, `git status --short` contains only intended files.
7. After commit, final `git status --short` is clean.
8. `origin/clinical-knowledge-architecture` is an ancestor of `HEAD`.

The safe push command is:

```bash
git push origin HEAD:clinical-knowledge-architecture
```

Force push is forbidden.

If the ancestry check fails, stop and report. Do not push.

If tests are skipped because of a known pre-existing environment timeout, the
final report must state that the full regression suite did not pass.

Future Codex prompts should include this rule so separate manual push prompts
are no longer needed.

The helper script for this workflow is:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/git_safe_push_detached_head_to_clinical_branch.ps1
```
