Instructions for Autonomous Coding Agent: Git Repository Cleanup
Objective: Systematically identify and rectify common sources of "messiness" in the current Git repository to improve its cleanliness, efficiency, and maintainability. Prioritize non-destructive operations.

Pre-computation/Assumptions:

The agent has already been working in this repository and understands its general purpose and typical build/artifact outputs.
The agent has necessary permissions to modify files and Git history (with extreme caution for history modification).
The current working directory is the root of the Git repository.
Phase 1: Assessment and Pre-Cleanup Safeguards
Status Check & Identify Messiness:

Execute git status --ignored and thoroughly analyze its output.
Untracked files/directories: Note all items listed under "Untracked files" and "Ignored files (if any, use -f to force add them)".
Changes not staged for commit: Note modified files.
Changes to be committed: Note staged files.
Execute git branch -va.
Local branches: Identify branches that are potentially merged into the current branch (e.g., git branch --merged).
Remote-tracking branches: Identify remote-tracking branches that no longer exist on the remote ([gone]).
Examine the .gitignore file:
Identify patterns that should be ignored but are missing (e.g., common build artifacts like dist/, build/, __pycache__, node_modules/, logs/, IDE files like .idea/, .vscode/, .DS_Store).
Identify patterns that are no longer relevant or overly broad.
Scan the file system for common temporary/artifact directories that are usually ignored (even if not currently untracked/listed by git status, e.g., temp/, cache/, _build/).
Safeguard Current Work:

Stash Uncommitted Changes: If git status shows "Changes not staged for commit" or "Changes to be committed", create a stash to preserve the current work.
Execute git stash push -m "Pre-cleanup stash for active work".
Confirm stash creation with git stash list.
Note: If the stash fails or the working directory cannot be cleaned due to complex conflicts, report this and seek human intervention.
Create a Temporary Cleanup Branch:

To ensure operations can be easily reverted, create a dedicated cleanup branch.
Execute git checkout -b cleanup-$(date +%Y%m%d%H%M%S).
Rationale: This allows for a safe space to perform operations, and the branch can be deleted if the cleanup is successful, or used to revert if issues arise.
Phase 2: Execution of Cleanup Operations (Non-Destructive First)
Clean Working Directory of Untracked Files/Directories:

Update .gitignore:
Add any identified missing patterns (from Phase 1, Step 1) to .gitignore. Ensure these patterns are precise.
Remove any redundant or incorrect patterns.
Commit these changes: git add .gitignore && git commit -m "chore: Update .gitignore for common artifacts"
Dry Run git clean:
Execute git clean -fdn (dry run for files and directories). Review the output carefully.
If the dry run lists items that should not be deleted, refine .gitignore again and re-commit.
Execute git clean:
If the dry run output is satisfactory, execute git clean -fd to remove untracked files and directories that are now properly ignored.
Rationale: This cleans temporary files, build artifacts, and other untracked junk without affecting committed history.
Local Branch Cleanup:

Delete Merged Local Branches:
Execute git branch --merged | grep -v "\*" | xargs -n 1 git branch -d.
Exclude: Do not delete main, master, develop, or the current HEAD branch.
Review the list before deletion. If xargs isn't available or direct deletion is preferred, list branches and delete them one by one.
Prune Stale Remote-Tracking Branches:
Execute git remote prune origin --dry-run and review the output.
If satisfactory, execute git remote prune origin.
Rationale: Removes references to branches that no longer exist on the remote.
Git Repository Optimization:

Garbage Collection:
Execute git gc --prune=now.
Rationale: This command cleans up unnecessary files and packs loose objects in the repository, making it more efficient and smaller.
Phase 3: Post-Cleanup & Verification
Verify Repository State:

Execute git status. Confirm there are no untracked files/directories (other than newly created artifacts, if applicable) and no unexpected uncommitted changes.
Execute git branch -va. Confirm only relevant local and remote-tracking branches remain.
Execute git log --oneline -5. Briefly inspect the recent commit history to ensure it looks as expected.
Re-apply Stashed Changes (if applicable):

If a stash was created in Phase 1, attempt to re-apply it.
Execute git stash pop.
Handle Conflicts: If conflicts arise during stash pop, report them and try to resolve them semi-autonomously using a standard strategy (e.g., preferring local changes, or reporting and seeking human guidance). If conflicts are too complex, revert git stash pop and report.
Integrate Cleanup Changes (Cushioned Commit):

If the cleanup was successful and the agent is on the cleanup- branch:
Execute git checkout <original_branch_name>.
Execute git merge cleanup-<timestamp_branch_name> --no-ff -m "chore: Repository cleanup" (or --squash if preferable, but --no-ff preserves the cleanup as a distinct merge).
Execute git branch -d cleanup-<timestamp_branch_name>.
If the cleanup was done directly on the original branch and new commits were made (e.g., .gitignore update), ensure these are pushed to the remote if relevant.
Final Report:

Summarize Actions Taken: List all Git commands executed for cleanup (e.g., git clean, git branch -d, git remote prune, git gc, .gitignore updates).
Report Before/After Metrics (if feasible):
Number of untracked files/directories removed.
Number of local branches deleted.
Repository size reduction (e.g., using du -sh .git/ before and after git gc).
Confirm Current Repository State: "Repository is now clean and optimized. All active changes have been reapplied."
Note Any Remaining Issues: "The following issues could not be fully resolved automatically: [list any conflicts during stash pop, or issues that required human intervention]."
Suggest Next Steps: "Consider reviewing the commit history for squashing or rebase operations if further history clean-up is desired (requires extreme caution)."
Important Considerations for the Agent:

Risk Assessment: Always prioritize safety. Operations like git clean -f and history rewriting (git rebase -i, git filter-repo) are powerful and potentially destructive. Avoid history rewriting unless explicitly tasked and with robust undo mechanisms.
Context Awareness: If the agent is aware of specific build systems (e.g., Python, Node.js, Java), it can use that knowledge to suggest more accurate .gitignore entries or temporary file locations.
Feedback Loop: If any step encounters an unexpected error or state, the agent should immediately pause, report the issue with full error logs, and await further instructions.
