# Production GUI Overhaul Design

Date: 2026-07-18
Status: Approved in conversation

## Goal

Turn the existing local NiceGUI application into a production-grade GIS and
research workstation without replacing the scientific services or changing the
toolkit's data contracts. The overhaul must improve visual hierarchy,
responsive behavior, workflow clarity, state correctness, and failure handling
across Scenarios, Configure, Candidates, Generate, Figures, and History.

This design supersedes the visual layout and mobile non-goal in
`2026-07-16-modern-gui-and-selection-workflow-design.md`. The product and
service boundaries from that document remain in force, as amended by the
current-only data model: NiceGUI, Leaflet, and Plotly remain; core workflows
remain local and offline; the GUI shares the package services used by the CLI;
the data catalog and DEM lifecycle remain CLI-only; candidate selection remains
manual and single-choice; and no compatibility layer is reintroduced.

## Verified Baseline

The current application was run from the repository's current `HEAD` on a
separate loopback port and exercised through the complete Chicago workflow:

```text
Scenarios
  -> Configure Chicago
  -> scan 23,192,505 grid positions
  -> inspect 19 candidates
  -> lock candidate 1
  -> generate five artifacts
  -> inspect History
  -> open the run in Figures
```

The real scan completed in approximately 8.2 seconds. The artifact generation
completed in approximately 8 seconds. No relevant browser console error was
reported during the exercised flow.

The run exposed visual and functional defects that a stylesheet-only refresh
would not fix:

- candidate-view controls display development option labels `A: Map` and
  `C: Map + Filmstrip` in both languages;
- the mobile top navigation overflows and clips destinations;
- the Configure page misplaces its saved-profile state and separates related
  actions across large empty regions;
- the top-bar job state is a static render-time snapshot and continues to say
  `No active job` while a scan or artifact generation is running;
- the application displays an unconditional `Ready` label even on unavailable
  and failed states;
- the candidate progress bar renders the raw numeric value `1` at completion;
- revisiting a completed candidate session exposes `Cache: none` and `0.0 s`
  instead of stable, meaningful scan provenance;
- raw internal values such as `ready`, `dem-pending`, `fast`, `preview_png`,
  `terrain_html`, and complete JSON objects appear in primary user-facing UI;
- Scenarios exposes separate Configure and Run buttons with the same target;
- History renders raw parameters and candidate JSON inline and provides no
  loading state while rebuilding its index;
- the standalone CLI web selector does not install the shared GUI stylesheet;
- an asynchronous online-map enable can win after the user has switched the
  layer off;
- Figures can retain source A after the user enters invalid source B; after B
  fails to load, the page still says the old rectangle is ready and leaves
  Export enabled, so the visible path and exported data can disagree.

## Chosen Approach

Use a systematic workbench overhaul. Preserve NiceGUI and the service layer,
but replace the shared shell, introduce a small set of reusable presentation
components, restructure each page around its real task, and repair the state
transitions that the rendered UI depends on.

A CSS-only facelift is rejected because it cannot repair stale controller
state, blank loading periods, duplicated actions, or asynchronous races. A new
frontend framework is rejected because it would duplicate or bridge the
existing Python UI layer, expand installation requirements, and create a much
larger regression surface without improving the scientific workflow.

## Design Direction

The visual language is a precise field-atlas workstation: calm, technical,
light, and map-led. It should feel like a purpose-built instrument rather than
a generic administration dashboard.

### Typography

- Display and navigation: `Bahnschrift`, `Aptos Display`, `Noto Sans`, then a
  sans-serif fallback.
- Body and controls: `Aptos`, `Segoe UI Variable`, `Noto Sans`, then a
  sans-serif fallback.
- IDs, coordinates, paths, timings, and commands: `Cascadia Code`,
  `SFMono-Regular`, then a monospace fallback.
- Type remains locally available. No remote font request or new runtime
  dependency is introduced.

### Color and material

The main tokens are defined once in `app.css` and used through semantic roles:

```text
canvas          #f2f1eb  warm atlas paper
surface         #fbfaf6  primary panel
surface-strong  #ffffff  raised work surface
frame           #0b3032  navigation and high-contrast framing
ink             #102d2e  primary text
ink-muted       #62716f  secondary text
accent          #149786  selected and actionable state
accent-soft     #d8eee8  low-emphasis selected state
signal          #c9772e  preparation, warning, and attention
success         #16745f  valid and completed state
danger          #b64141  destructive and failed state
border          #d6ddd8  quiet structure
focus           #36b8a7  keyboard focus
```

Cards use fine borders and restrained shadows. A subtle coordinate-grid or
contour-line treatment may appear in page headers, but it must never compete
with the actual map or reduce text contrast. Gradients are limited to small
status accents and map-adjacent atmosphere; no purple-on-white marketing
gradient is used.

### Motion

Motion communicates state rather than decorating every control:

- the application shell and first content group may enter with one short,
  staggered reveal;
- progress changes, selected candidates, drawers, and inline errors transition
  smoothly within 120-180 ms;
- skeletons replace blank waits;
- reduced-motion preferences disable non-essential movement.

## Information Architecture

### Desktop shell

At widths of 981 px and above, use a persistent 224 px navigation rail and a
compact top command bar.

The rail contains the product identity and the four persistent destinations:
Scenarios, Configure, Figures, and History. Candidate and Generate remain
session-bound workflow pages; when active, a compact workflow indicator in the
page header shows Scenarios -> Configure -> Candidates -> Generate.

The top command bar contains the current page context, a live active-job
indicator, and language selection. The unconditional application `Ready` label
is removed. When no job is active, the job control is visually quiet. When a
job is active, it names the operation and exposes progress or phase when
available.

Main content uses page-specific widths. Data-entry and history pages remain
bounded for readability. Candidate Explorer is allowed to use the full
available workspace.

### Tablet and mobile shell

At widths of 980 px and below, the rail becomes a modal drawer opened from a
stable menu button. The top command bar never scrolls horizontally. At widths
of 760 px and below, page actions stack or move into a sticky bottom action
region, map tools wrap into two rows, and all touch targets are at least 44 px.

No route may require horizontal page scrolling at 390 px. Data tables,
coordinate groups, and command snippets may scroll inside their own bounded
regions when necessary.

## Shared Presentation Components

Add framework-light presentation helpers under `gui/` rather than duplicating
page markup:

- `render_page_header`: title, description, workflow context, and page actions;
- `render_status_badge`: localized label, semantic color, and optional machine
  value inside technical details only;
- `render_action_bar`: consistent primary, secondary, and destructive action
  order, including sticky behavior where appropriate;
- `render_empty_state` and `render_loading_state`: intentional unavailable,
  empty, and waiting surfaces;
- `render_technical_details`: collapsed diagnostics, paths, raw identifiers,
  codes, and JSON;
- `render_job_indicator`: a live view of the process-local coordinator;
- presentation mappings for readiness, scan mode, cache state, artifact state,
  and run state in both languages.

These helpers receive state and callbacks. They do not read the catalog,
perform scientific work, or own persistence.

## Page Designs

### Scenarios

The landing header shows total, ready, and preparation-required counts. Each
scenario card prioritizes the display name, a localized readiness badge, a
short readiness explanation, and one primary next action.

- A ready scenario exposes `Configure and scan` as its primary action.
- A non-ready scenario exposes `View setup guidance`; it does not present a
  disabled Run button as if execution were almost possible.
- Fast Validate is secondary. Full Checksum is placed in an overflow or
  explicitly slow action area.
- Dataset IDs, manifest details, exact paths, validation codes, and CLI commands
  are grouped below expandable technical or setup details.
- The separate Configure and Run buttons are removed.

Global CLI guidance remains available, but scenario-specific commands are shown
next to the scenario that needs them. Placeholder paths are labeled as values
the user must replace, not as directly runnable commands.

### Configure

The page becomes a configuration workbench:

- a compact profile header contains the profile selector, saved/draft/dirty
  state, and profile-management menu;
- the main form uses a stable two-column grid on desktop and one column on
  mobile;
- basic fields remain visible and advanced fields remain collapsible;
- output destination is a clearly separated final form group;
- Copy, Rename, Set Default, and Delete live in one profile-management menu or
  tightly grouped secondary area;
- Discard, Save, and Start Scan share one ordered action bar, with Start Scan as
  the single primary action;
- destructive confirmation text describes the concrete consequence rather than
  referring generically to a repository change.

The saved-profile label cannot float independently from the selector. Dirty and
validation state remain visible until explicitly resolved.

### Candidate Explorer

The candidate page is the visual center of the product.

- The shared shell and a map skeleton render immediately while offline assets
  are prepared.
- The header contains a segmented `Map` / `Map and candidates` control with no
  development option letters.
- The scan panel shows a human percentage, checked/total positions, candidate
  count, elapsed time, and localized cache provenance. At completion it may
  collapse into a compact summary strip.
- Scan lifecycle controls, candidate navigation, and candidate confirmation are
  visually separated instead of forming one undifferentiated button row.
- Layer switches, DEM opacity, and DEM style occupy a compact map toolbar.
- On desktop, the map uses the main work area and the inspector uses a stable
  320-360 px side panel. The map fits the boundary without leaving most of the
  canvas as empty gray space.
- `Map and candidates` adds a bounded horizontal filmstrip below the map rather
  than turning the page into a long five-column gallery. Selection, map bounds,
  and layer state persist across modes.
- Candidate cards show the candidate number, station count, and a readable
  terrain preview. The raw flat-grid ID remains available as secondary
  technical data.
- The inspector leads with terrain statistics and station count. Coordinates,
  bounds, scan mode, and pixel counts move into a technical-details section.
- Mobile places the inspector below the map and keeps the filmstrip
  horizontally scrollable.

Confirm remains unavailable until the scan is complete and one candidate is
selected. The selected card and inspector use one consistent interface accent
state. Map rectangles retain the Canvas Station Rendering contract: unselected
rectangles remain red, the selected rectangle remains green, and rectangles
remain visually above individually clickable station dots.

### Generate

The locked-selection summary becomes a compact, human-readable review card.
The artifact section becomes a list of named deliverables with short
descriptions, format badges, selection controls, and live state icons.

Machine keys such as `preview_png` and `terrain_html` never appear in primary
copy. During generation, the page shows the active artifact or phase and the
top command bar also reflects the active job. Generate is the primary action;
Open in Figures is secondary and clearly explains whether it uses the current
locked selection or a published run.

### Figures

Figures uses a source -> preview -> style -> export composition. Its source
contract remains exactly the current-only contract: the current confirmed
selection, or a completed toolkit run directory or its `run.json`. A bare CSV,
DEM attachment, or compatibility source is neither accepted nor displayed.

The page composition is:

- source selection spans the top and shows the loaded source separately from
  the editable path;
- the central preview surface receives most horizontal space;
- style controls occupy a structured side panel on desktop and collapsible
  groups on mobile;
- final format selection and destination form a concise export summary;
- long paths use monospace text with copy affordances and safe wrapping.

Source state is fail-closed. Editing the source field immediately marks the
loaded source dirty and disables Preview and Export. A successful Load replaces
the controller source and clears the dirty state. A failed Load leaves no
exportable source associated with the edited field. The page must never display
source B while exporting source A.

Changing a style field marks the preview stale. Stale is shown as a status
badge beside Preview rather than as an isolated line of text.

### History

The shell and a loading skeleton render before run discovery begins. Completed
discovery replaces the skeleton with compact run cards or an intentional empty
state.

Each card leads with scenario/profile, human-readable time, run status, artifact
count, and parent relationship. Primary actions are Inspect and Open in
Figures. Reveal Directory is secondary. The only repair action is `Retry
missing terrain figures`, and it appears only for a current run whose own
current metadata records missing expected terrain outputs. It never infers
artifacts from a legacy or invalid record. Raw run IDs, parameters, candidate
coordinates, artifact filenames, and JSON move into a collapsed
technical-details region.

## Functional Corrections

### Live job state

The shell must not receive a one-time string snapshot. A lightweight NiceGUI
timer or shared observable polls the coordinator at a restrained interval and
updates the job indicator without navigation. The state returns to idle when
the job completes or fails. The indicator is display-only and does not become
a second job coordinator.

### Candidate session provenance

A completed session retains the authoritative elapsed time, cache hit/miss
state, checked positions, and result count. Re-rendering the page derives its
summary from the completed session/result rather than a new default controller.
Presentation mappings convert `none`, `hit`, and `miss` into user-facing copy.

### Online layer race

Each online-layer request carries a monotonically increasing intent token or
checks the current desired switch value after the awaited probe. A stale enable
completion cannot add a layer after the user has switched it off. Disposing the
page invalidates pending intent.

### Standalone selector styling

The CLI web selector loads the same packaged stylesheet and local Leaflet
assets as the main application. It reuses the Candidate presentation without
requiring the full navigation shell.

### History loading

History renders the shell and loading state synchronously, waits for discovery
off the event loop, then replaces only the content region. Discovery errors
produce an inline summary and expandable technical detail.

### Error presentation

Primary error copy answers what failed and what the user can do next. Machine
codes, exception text, raw paths, and JSON remain available inside technical
details. Toasts reinforce an error but do not become the only durable evidence
of failure.

## Accessibility and Responsive Contract

- Keyboard focus is visible on every interactive control.
- Navigation, segmented controls, dialogs, and expandable details expose
  correct labels and selected/expanded state.
- Status is never communicated by color alone.
- Normal text and controls meet WCAG AA contrast.
- Touch targets are at least 44 px at narrow viewports.
- Reduced-motion preferences disable non-essential animation.
- The app is verified at 1440 x 1000, 1024 x 768, 981 px, 980 px, 761 px,
  760 px, and 390 x 844 so both responsive boundaries are exercised.
- At 390 px, the application has no clipped navigation, overlapping controls,
  off-screen dialogs, or page-level horizontal scrollbar.

## Testing Strategy

Behavior changes follow red-green-refactor. Regression tests are written and
observed failing before production changes.

### Focused regressions

- translation and rendered-copy tests reject `A:`, `B:`, and `C:` option
  prefixes and verify localized readiness, cache, scan-mode, artifact, and run
  status mappings;
- Figures test: load valid A, edit to invalid B, fail loading B, and prove that
  Preview and Export cannot use A;
- job-indicator test: idle -> delayed active job -> completed/failed -> idle
  without navigation;
- online-layer test: enable, disable before a delayed probe resolves, then prove
  that no tile layer is added;
- standalone-selector test proves the packaged CSS and local map resources are
  registered;
- source-contract tests prove Figures accepts only the current confirmed
  selection or a completed current run directory/`run.json`, rejects bare CSV,
  and exposes no Attach DEM or compatibility-source controls;
- History test proves that shell/loading content renders before a delayed index
  rebuild completes;
- completed-session test proves elapsed time and cache provenance survive a
  route re-render;
- progress presentation test proves completion displays `100%`, not `1`;
- page tests prove duplicate scenario actions and raw machine-token status rows
  are absent;
- map-rendering tests preserve one interactive canvas-backed circle per station,
  no DOM marker per station, rectangle-above-station ordering, existing
  red/green rectangle colors, and identical GUI/standalone-selector behavior.

### Rendered browser verification

Run the current application through a real Chromium engine and capture before
and after evidence. For every primary page:

- verify URL and title;
- verify meaningful DOM content and absence of a framework error overlay;
- collect browser errors and warnings;
- exercise at least one relevant interaction;
- inspect desktop, tablet, and mobile viewport screenshots;
- check clipping, overlap, wrapping, scroll traps, disabled states, focus,
  loading, empty, success, and failure states.

The full Chicago path is repeated after implementation. Figures explicitly
repeats the stale-source reproduction. Candidate Explorer verifies selection,
filmstrip mode, station rendering, and the online toggle race.

### Repository gates

```powershell
python -m ruff check src scripts tests
python -m compileall -q src/lte_scenario_toolkit scripts
python -m pytest -q
git diff --check
```

No test, screenshot, trace, or temporary browser script is committed as a
product artifact unless it is a durable regression fixture.

## Non-Goals

- replacing NiceGUI, Leaflet, Plotly, or the Python-only installation model;
- changing candidate validity, ordering, scanning algorithms, or scientific
  outputs;
- adding automatic candidate ranking or recommendation;
- adding multi-selection, freehand geometry, or parameter sweeps;
- moving scenario registration, Earth Engine export, or DEM ingest into the
  GUI;
- adding authentication, a server deployment model, or an application
  database;
- introducing a remote font, tile, analytics, or asset dependency;
- reintroducing legacy profile, cache, selector, CSV-source, or run-record
  compatibility removed by the current-only model;
- broad service refactoring unrelated to a verified GUI behavior.

## Implementation Boundaries

Reusable visual composition belongs under `src/lte_scenario_toolkit/gui/`.
Page modules remain adapters around services. Scientific computations, path
rules, run publication, profile persistence, and data validation remain in
their existing package services.

The expected production change surface is:

- `gui/layout.py` and a small shared presentation module;
- `gui/assets/app.css` and existing local map assets where required;
- `gui/i18n.py` for complete English/Chinese presentation mappings;
- the six page modules for layout and state corrections;
- `gui/app.py` for live shell state and loading-first route composition;
- `web_selector.py` for shared style/resource registration;
- focused tests in the existing GUI, selector, and service test modules.

The implementation must preserve the current public CLI commands, file
contracts, scenario catalog, profiles, and generated run structure.

## Implementation Order

1. Add failing regression tests for the reproduced functional defects and
   developer-residue copy.
2. Introduce design tokens, shell primitives, localized presentation mappings,
   and responsive navigation.
3. Repair live job, Figures source, online-layer, History loading, completed
   session, and standalone-selector state behavior.
4. Recompose Scenarios and Configure around the shared components.
5. Recompose Candidate Explorer and verify the real Chicago map/scan path.
6. Recompose Generate, Figures, and History in populated and empty states.
7. Run focused tests, the full repository gates, and the complete rendered
   browser QA matrix.

## Acceptance Criteria

The overhaul is complete when all of the following are true:

1. No user-facing control contains development option prefixes such as `A:` or
   `C:`.
2. The six primary page surfaces use one coherent workstation shell and visual
   system.
3. Navigation and primary interactions work without clipping or page-level
   horizontal scrolling at 390 px.
4. A running scan or generation job is reflected live in the shared shell and
   returns to idle without a reload.
5. Candidate progress, elapsed time, cache provenance, scan mode, and selection
   details are stable and human-readable after re-entry.
6. Figures cannot export a stale source after the source field changes or a new
   load fails.
7. A stale online-layer request cannot override the latest user intent.
8. History shows a shell and loading state before discovery finishes.
9. The standalone CLI web selector receives the shared candidate layout and map
   sizing styles.
10. Station rendering remains canvas-backed and individually clickable, creates
    no DOM marker per station, preserves rectangle-above-station ordering and
    red/green rectangle states, and behaves identically in the GUI and
    standalone selector.
11. Figures exposes only a confirmed current selection or completed current run
    directory/`run.json`; bare CSV and Attach DEM controls remain absent.
12. Raw machine keys and JSON are absent from primary user-facing status and
    summary regions but remain available in technical details where useful.
13. Empty, loading, success, partial, failed, disabled, and unavailable states
    are visually intentional and actionable.
14. The full Chicago workflow and Figures failure reproduction pass in a real
    browser with no relevant console errors.
15. Focused tests, the complete Pytest suite, Ruff, compile checks, and
    `git diff --check` pass.
