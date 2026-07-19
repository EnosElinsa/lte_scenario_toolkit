# Run History Transactional Trash Design

Date: 2026-07-19
Status: Approved in conversation

## Goal

Add safe run deletion to the local NiceGUI History workflow without weakening
run provenance, path containment, or reproducibility. A normal delete moves one
run or one complete derived run family into an application-managed trash. The
trash supports whole-family restore and explicitly confirmed permanent deletion.
Trash contents never expire automatically.

The implementation must preserve the current run manifest format and immutable
published-run contract. It must not leave a visible child run whose parent or
recorded source was deleted by the application.

## Verified Baseline

History currently exposes only deliberately non-destructive actions:
`Inspect`, `Open in Figures`, `Retry missing`, `Reveal Directory`, and page
refresh. `HistoryAction` documents this boundary, and a regression test asserts
that no delete action exists.

History is reconstructed from live `run.json` files across the repository
results directory and configured output roots. Its cache index is explicitly
non-authoritative. Every action re-discovers and revalidates the selected run by
root, path, run ID, scenario ID, profile ID, and creation time.

Published selection and figure runs form a provenance graph through both
`parent_run_id` and `metadata.source`. A completed figure run normally carries
its own `source.csv`, but its parent and source references still matter for
traceability. Partial derived runs may require their recorded source to retry
missing artifacts.

At design time, the current local results contain eight published runs:

- three selection runs totaling approximately 345.3 MiB;
- five figure runs totaling approximately 451.2 MiB;
- approximately 796.5 MiB in total.

The current History action row already contains three or four visible buttons,
so a destructive action must not become another visually equal primary button.

## Product Decisions

The following decisions are approved:

1. `Delete` means move to the application trash, not hard-delete.
2. Trash items are retained until the user permanently deletes them.
3. No time-based expiration or background purge is allowed.
4. Deleting a run with direct or indirect dependents moves the complete
   transitive run family.
5. The application never offers a mode that deliberately leaves child runs
   orphaned.
6. A run family remains indivisible in trash: restore and permanent deletion
   operate on the whole family.
7. The first version has no bulk selection and no `Empty trash` action.

## Alternatives Considered

### Application-managed transactional trash — chosen

Move validated run directories into a reserved trash directory below each
output root, record the operation in a transaction manifest, and expose restore
and purge in History. This is cross-platform, auditable, and can preserve the
run dependency contract. It requires explicit transaction and recovery logic.

### Operating-system recycle bin — rejected

The application cannot reliably enumerate and restore one logical run family
from the Windows, Linux, and macOS recycle-bin implementations. Cross-root
families cannot be treated as one operation, and the resulting behavior would
depend on the host shell.

### Hidden or tombstoned live runs — rejected

Hiding a directory only in the derived History index would not work because a
fresh live discovery would add it again. Mutating an immutable `run.json` to
mark it deleted would weaken the current publication contract and would not
release storage.

### Immediate permanent deletion — rejected as the default

It releases disk space immediately but provides no recovery from an accidental
selection and does not match the approved product behavior.

## Terminology and Identity

A **live run** is one path-validated `RunEntry` currently discoverable by
`RunService` below a configured output root.

A **run identity** is the tuple:

```text
(canonical output root, run_id, canonical expected run path)
```

Run ID alone is never sufficient because History supports multiple output
roots and must tolerate an ID collision or copied run tree safely.

A **dependent** is a live run that refers to another live run through either:

- `parent_run_id`; or
- the paired `metadata.source.run_id` and `metadata.source.path` fields.

A **run family** is the selected run plus the complete transitive closure of
its dependents. A leaf run therefore forms a family of one.

A **trash transaction** is one user-approved move of one run family. It has one
globally unique transaction ID even when the family spans multiple output
roots.

## User Experience

### History action

Each published run card gains `Move to Trash` in a trailing overflow menu. It
uses the danger semantic color and remains visually separated from `Inspect`
and `Open in Figures`.

Selecting it performs a read-only fresh plan before opening a confirmation
dialog. The dialog displays:

- scenario and profile;
- local creation time and the first eight characters of the run ID;
- run kind and status;
- artifact count;
- total recorded disk size;
- direct and indirect dependent count;
- every affected output root;
- an expandable list of all affected runs.

For a leaf, the primary action reads `Move run to Trash`. For a parent, it
reads `Move {count} related runs to Trash`, and the consequence text states
that all derived runs move together. There is no `Delete only this parent`
choice.

Opening the dialog does not reserve the plan indefinitely. Confirming it
rebuilds the graph and compares the new plan with the displayed plan. A changed
lineage, path, run set, or manifest identity closes no gap: the operation stops
and asks the user to refresh and review the new impact.

After success, History is rebuilt in place and shows a positive notification.
The run family disappears from normal History immediately. Storage is not
reported as reclaimed until permanent deletion succeeds.

### Trash surface

The History header adds `Trash ({count})` beside Refresh. It opens
`/history/trash`, which remains under the History navigation section and uses
the same shell and loading-first behavior.

Trash displays one card per logical transaction, not one card per run. A card
shows:

- deletion time;
- original scenario/profile summary;
- transaction ID prefix;
- number of runs and artifacts;
- total size at deletion time;
- output roots involved;
- transaction health.

Technical details expose the original relative paths, run IDs, graph edges,
and transaction journal without placing raw JSON in the primary UI.

The two primary item actions are:

- `Restore run family`;
- `Delete permanently`.

Restore is unavailable when the transaction is incomplete, a destination path
is occupied, an output root is unavailable, or a family member is in use.

A `recovery_required` card exposes only `Recover transaction`. Recovery uses
the journal to return an interrupted move to its fully live state or an
interrupted restore to its fully trashed state. It never chooses between two
occupied copies. A `purge_failed` card exposes only `Retry permanent deletion`
because any payload deletion permanently disables restore.

Permanent deletion opens a second danger confirmation. The user must enter the
first eight characters of the transaction ID. The dialog states the number of
runs, total size, and that restoration becomes impossible as soon as purge
starts.

There is no automatic retention timer, bulk selection, or `Empty trash` action
in the first version.

### Failure presentation

Primary messages explain the action the user can take. Raw exceptions, paths,
and transaction details stay in collapsed technical details.

Examples include:

- `This run family changed. Refresh History and review the updated impact.`
- `This run is open in Figures. Close it or choose another source first.`
- `The original path is occupied, so this family cannot be restored.`
- `Permanent deletion stopped before all trash data was removed. Retry the
  deletion; this transaction can no longer be restored.`

## Architecture

The feature is separated into storage lifecycle, graph/transaction logic, and
GUI adaptation.

### `run_service.py`: root-local filesystem authority

`RunService` remains the authority for paths below one output root. It gains
public root-local primitives for:

- validating a live entry immediately before mutation;
- creating and validating the reserved `.trash` directory;
- moving one live run into one prepared transaction directory;
- moving one trashed run back to its exact expected live path;
- validating and permanently removing one transaction portion;
- discovering root-local transaction manifests without traversing payload
  artifacts.

These primitives reuse the existing lexical containment, canonical
containment, symlink, junction, and Windows reparse-point protections. They do
not build cross-root dependency graphs and do not render UI.

Normal run discovery must skip `.trash` entirely. Trash discovery only reads
direct transaction manifests below `.trash`; it does not accept arbitrary
paths supplied by the browser.

### `run_trash.py`: graph and transaction orchestration

A new GUI-independent module owns:

- `RunIdentity`;
- `RunDependencyGraph`;
- `TrashPlan`;
- `TrashTransaction` and root-local transaction portions;
- multi-root move, rollback, restore, and permanent-delete orchestration;
- process-local run usage leases.

The planner consumes fresh `RunEntry` discoveries for all configured History
roots. Edges use path plus ID and reject ambiguous source references. It
computes the transitive dependent closure and a stable plan fingerprint from
the selected identity, affected identities, graph edges, and immutable manifest
identity fields.

If a dependency points outside the configured roots, is ambiguous, or cannot
be validated, a parent-family move is blocked instead of guessing. An already
orphaned run may still be moved with its own discoverable dependents; the
feature does not rewrite historical manifests to repair pre-existing damage.

### `gui/pages/history.py`: presentation adapter

History adds the new action models, confirmation dialogs, Trash cards, and
localized error presentation. It receives callbacks and immutable plan or
transaction snapshots. It does not call `shutil.rmtree`, construct untrusted
paths, or own transaction state.

### `gui/app.py`: application orchestration

The app owns the shared trash service, configured-root provider, coordinator
check, and asynchronous callbacks. Move, restore, and purge run with NiceGUI's
I/O-bound execution path, followed by a fresh History or Trash rebuild only if
the client remains alive.

The shared job indicator gains distinct kinds for trash move, restore, and
permanent deletion. Destructive operations are unavailable while any existing
scan, generation, figure, validation, or trash job is active.

### Figures usage leases

An idle page can otherwise keep a run loaded while another tab moves it. Each
Figures controller therefore acquires leases for its loaded run source and
explicit parent run. It releases them when the source changes, the page closes,
or the controller is disposed.

A Trash plan whose family intersects an active lease is rejected before any
filesystem change. External programs cannot participate in this process-local
registry; a filesystem sharing or lock failure remains a safe operation error.

Route-request tokens do not hold leases by themselves. If a route is opened
after its run moved to Trash, existing fresh source validation produces an
unavailable-source message instead of resurrecting or using stale data.

## Trash Storage Contract

Each affected output root stores its portion at:

```text
<output-root>/.trash/<transaction-id>/
  trash.json
  runs/<scenario-id>/<profile-id>/<original-run-directory>/...
```

Every root-local `trash.json` uses one schema version and records:

- `schema_version`;
- `transaction_id` and shared logical group ID;
- `deleted_at`;
- current state;
- all roots expected in the logical transaction;
- selected run identity;
- root-local family members;
- original and trash-relative paths;
- run IDs, parent IDs, and resolved source edges;
- artifact counts and sizes captured during planning;
- completed move or restore steps;
- failure details needed for recovery.

Size calculation walks only validated real directories and regular files. It
does not follow a symbolic link, junction, reparse point, or redirected
artifact.

The manifest is written atomically and updated after each completed rename.
No absolute browser-supplied destination is persisted or trusted. Original
paths are stored as validated root-relative paths and reconstructed below the
recorded canonical root.

Trash directories and transaction directories must be real directories. A
symlink, junction, reparse point, lexical traversal, canonical escape, reserved
device path, or path mismatch makes the transaction non-actionable.

## Dependency Rules

The graph is rebuilt across every current History output root.

The no-orphan guarantee applies to the complete set of output roots configured
and currently discoverable by this GUI. A run copy or dependent stored in an
unknown, unconfigured external root cannot be discovered or coordinated; the
confirmation identifies the roots that form the deletion universe.

For `parent_run_id`, a parent is resolved within the same canonical output root
using the parent ID and expected run shape.

For `metadata.source`, both the recorded source path and source run ID must
match one live identity. Matching only one field is insufficient.

Deletion follows reverse edges from the selected run and includes every direct
and indirect dependent. A child may be moved without its parent only when the
parent remains live. A parent cannot move without all of its descendants.

The rule applies even when a completed figure run contains a self-contained
`source.csv`: provenance is part of the run contract, not only a runtime input.

Restore and permanent delete operate on the exact family captured in the
transaction. The application never restores or purges a subset of that family.

## Move Transaction

Move-to-trash follows this sequence:

1. Re-discover all roots and build the dependency graph.
2. Resolve the clicked immutable `HistoryRunReference` against live state.
3. Compute the complete family, affected roots, size, and plan fingerprint.
4. Reject an active shared job or intersecting run usage lease.
5. Present the plan and receive explicit confirmation.
6. Repeat steps 1-4 and require the fingerprint to match.
7. Preflight every root, trash path, run path, permission boundary, and
   transaction destination before moving any run.
8. Write a `moving` transaction manifest to every affected root.
9. Move descendants before their parents. Each same-root directory rename is
   atomic and is journaled immediately.
10. If any move fails, move every completed item back in reverse order.
11. Mark every root-local portion `trashed` only after all family members move.
12. Rebuild History and Trash from live filesystem state.

Creating a cross-root family uses one logical transaction ID and one manifest
portion per root. If any root is unavailable or unwritable during preflight,
nothing moves.

A handled failure that rolls back completely leaves the original History
unchanged. If rollback itself fails or the process stops mid-transaction, the
journal remains discoverable as `recovery_required`; neither normal History
nor Trash offers restore or purge until recovery reconciles every recorded
path.

After a successful move, empty profile or scenario parents may be removed only
with non-recursive `rmdir` calls. Failure to remove an empty parent is harmless
and never changes transaction success.

## Restore Transaction

Restore follows this sequence:

1. Discover every root-local portion with the shared transaction ID.
2. Require all portions to be healthy and in `trashed` state.
3. Reconstruct and validate every original path below its canonical root.
4. Require every original destination to be absent; restoration never
   overwrites a live run or directory.
5. Reject an active job or conflicting usage lease.
6. Mark every portion `restoring`.
7. Recreate only validated real parent directories.
8. Restore parents before descendants and journal each atomic rename.
9. On a handled failure, move restored members back to Trash in reverse order.
10. Remove transaction directories only after the entire family is live and
    freshly discoverable with matching identities.
11. Rebuild History and Trash.

If restore rollback fails, the transaction becomes `recovery_required` and is
not eligible for permanent deletion until reconciled.

## Permanent Deletion

Permanent deletion is available only for a complete transaction in `trashed`
state and only after the transaction-ID confirmation.

Before deletion, every root and transaction path is revalidated. The recursive
delete target must be exactly a non-redirected
`<canonical-root>/.trash/<validated-transaction-id>` directory. The output root,
`.trash` root, arbitrary descendants, and user-entered paths are never valid
recursive-delete targets.

All portions are marked `purging` before data removal begins. Cancellation is
not offered once purge starts because partial cancellation cannot restore
deleted files safely.

If every portion is removed successfully, the transaction disappears and the
reported size is reclaimed. If any portion fails after deletion has begun, the
remaining journals are marked `purge_failed`. A `purge_failed` transaction can
only retry permanent deletion; restore is permanently disabled.

No scheduler, startup hook, retention age, or background task automatically
purges Trash.

## Concurrency and Stale-State Handling

The shared `JobCoordinator` remains the application-wide mutation gate. Only
one scan, generation, figure, validation, trash, restore, or purge job may be
active.

Every destructive confirmation is based on a fresh plan and is validated again
at click time. The operation fails closed when:

- a run disappeared or moved;
- a manifest identity or dependency changed;
- a new dependent appeared;
- an output root became unavailable;
- a run became leased by Figures;
- a transaction destination appeared;
- an original restore destination appeared.

Other open History tabs may hold stale cards. Their callbacks already use live
reference validation; a moved run produces the existing refresh-required
behavior rather than acting on a stale path.

External CLI processes are not coordinated by the GUI job or lease registries.
Atomic rename and filesystem errors therefore remain the final guard. Sharing
violations and permission failures are reported without falling back to copy
and delete.

## Index and Discovery Behavior

The existing History index remains a derived cache. After move, restore, or
purge, the UI rebuilds from live run and trash manifests and then rewrites the
cache. A cache-write failure produces a diagnostic but does not undo a
successful filesystem transaction.

Normal History never discovers `.trash` payloads. Trash discovery never treats
a payload `run.json` as a live run. An incomplete or invalid transaction
manifest appears as a Trash diagnostic and cannot be acted upon through normal
buttons.

## Internationalization and Accessibility

All action, confirmation, consequence, failure, recovery, and status copy is
provided in English and Chinese. The words `Trash`, `Restore`, and `Delete
permanently` are distinct; the UI never labels a reversible move as immediate
permanent deletion.

Danger is not communicated by color alone. Dialog titles and consequence text
name the action, run count, and reversibility. Keyboard focus moves into dialogs
and returns to the invoking control. Buttons maintain the existing minimum
touch targets and responsive stacking behavior.

## Testing Strategy

All behavior changes follow red-green-refactor. Tests use temporary output
roots and small synthetic artifacts; they never mutate the user's real
`results/` directory.

### Graph and planning tests

- leaf family contains exactly one run;
- direct and multi-level descendants form one transitive family;
- both `parent_run_id` and source path plus source ID create edges;
- completed figure snapshots still preserve provenance edges;
- cross-root sources use canonical root, path, and ID without collision;
- ambiguous, missing, or malformed references block a parent-family plan;
- a newly published child invalidates a previously displayed plan;
- pre-existing orphan handling does not rewrite manifests.

### Filesystem transaction tests

- leaf and family moves preserve every byte and original relative path;
- descendants move before parents and restore after parents;
- normal discovery skips `.trash`;
- Trash discovery merges root-local portions by transaction ID;
- a failure at every move step rolls back earlier moves;
- simulated process interruption leaves a recoverable journal;
- restore preflight rejects any occupied destination before moving data;
- restore failure rolls back to a complete trashed family;
- cross-root preflight failure moves nothing;
- permanent deletion rejects live, root, traversal, symlink, junction, and
  reparse-point targets;
- partial permanent deletion becomes `purge_failed` and cannot restore;
- no code path purges according to age or startup time.

### Concurrency tests

- any active shared job disables and rejects destructive actions;
- a Figures source or parent lease blocks a family plan;
- releasing or changing the Figures source releases the lease;
- stale History cards fail against fresh discovery;
- external sharing and permission failures preserve the recoverable state.

### GUI tests

- History shows `Move to Trash` only for published live runs;
- pending confirmed selections do not expose run deletion;
- a leaf dialog shows one run and its size;
- a parent dialog names the complete family and offers no orphaning choice;
- success refreshes History and updates the Trash count;
- Trash loading, empty, healthy, recovery-required, and purge-failed states are
  intentional;
- restore and permanent-delete controls follow transaction health;
- permanent deletion requires the transaction ID prefix;
- English and Chinese copy describes reversible and irreversible actions
  correctly;
- primary errors remain visible with technical details available separately.

### Rendered browser verification

Exercise the local application through a real browser:

1. move a leaf figure run to Trash and verify it leaves normal History;
2. restore it and verify all existing actions work again;
3. move a selection run and its derived figure runs as one family;
4. verify no child remains in normal History;
5. restore the family and verify parent-before-child lineage;
6. move a disposable synthetic family and permanently delete it after the ID
   confirmation;
7. verify active-job and open-Figures lease blocking;
8. check desktop and narrow layouts, dialogs, focus, console health, and stale
   multi-tab behavior.

Real browser verification uses disposable test output roots for destructive
flows. The existing user results are never used as delete fixtures.

### Repository gates

```powershell
python -m ruff check .
python -m compileall -q src
python -m pytest -q
git diff --check
```

## Failure Recovery Matrix

| Phase | Failure outcome | User action |
|---|---|---|
| Planning/preflight | No files moved | Refresh and retry |
| Moving with successful rollback | Original family remains live | Retry |
| Moving with failed rollback or crash | `recovery_required` | Run recovery |
| Restoring with successful rollback | Complete family remains in Trash | Retry |
| Restoring with failed rollback or crash | `recovery_required` | Run recovery |
| Purging before any payload deletion | Complete Trash item remains | Retry purge |
| Purging after any payload deletion | `purge_failed`, restore disabled | Retry purge |
| History-index rewrite | Filesystem result remains authoritative | Refresh; inspect diagnostic |

Recovery is deterministic from the journals and current paths. It never guesses
which copy is authoritative and never overwrites an occupied destination.

## Non-Goals

- automatic Trash expiration or retention scheduling;
- OS recycle-bin integration;
- bulk run selection or `Empty trash`;
- restoring or purging only part of one run family;
- rewriting child manifests to detach them from a deleted parent;
- deleting in-memory pending candidate selections through this feature;
- deleting output roots, profiles, source DEMs, boundaries, or catalog data;
- deleting arbitrary paths pasted by the user;
- coordinating with remote machines or external CLI processes beyond safe
  filesystem failure;
- changing the current run manifest schema or scientific artifacts.

## Expected Change Surface

- `src/lte_scenario_toolkit/run_service.py`: root-local trash primitives and
  discovery exclusions;
- `src/lte_scenario_toolkit/run_trash.py`: dependency graph, plans,
  transactions, recovery, and leases;
- `src/lte_scenario_toolkit/gui/pages/history.py`: actions, dialogs, and Trash
  presentation;
- `src/lte_scenario_toolkit/gui/pages/figures.py`: run usage leases;
- `src/lte_scenario_toolkit/gui/app.py`: shared services, routes, async
  orchestration, and refresh;
- `src/lte_scenario_toolkit/gui/i18n.py`: complete English and Chinese copy;
- `src/lte_scenario_toolkit/gui/presentation.py`: trash and job-state mappings;
- focused service and GUI tests, with browser evidence stored outside the
  repository.

## Acceptance Criteria

1. A leaf run can move to Trash and disappear from normal History without data
   loss.
2. Selecting a parent always includes all direct and indirect dependents; no
   orphaning option exists.
3. One Trash card represents the complete family across all involved roots.
4. A healthy family restores only when every original destination is free, and
   all runs return with identical artifacts and manifests.
5. Permanent deletion requires transaction-ID confirmation, frees the Trash
   payload, and cannot target a live or unvalidated path.
6. Trash data is never deleted automatically.
7. Active jobs and open Figures leases block conflicting mutations.
8. Stale plans, root changes, new dependents, and permission failures stop
   before partial mutation or roll back safely.
9. Interrupted move or restore transactions are discoverable and recoverable;
   interrupted permanent deletion is retryable but never restorable.
10. Normal History never discovers Trash payloads, and the derived cache never
    becomes authoritative.
11. English and Chinese interfaces distinguish move, restore, and permanent
    deletion and explain the affected family.
12. Focused tests, the full test suite, Ruff, compilation, diff checks, and the
    rendered disposable-data browser workflow pass.
