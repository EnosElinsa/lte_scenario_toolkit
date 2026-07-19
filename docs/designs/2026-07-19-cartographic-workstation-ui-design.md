# Cartographic Workstation UI Design

Date: 2026-07-19
Status: Approved in conversation

## Goal

Redesign the local NiceGUI application as a cohesive cartographic workstation.
The redesign must make local geography the visual focus, remove crowded button
rows, provide an explicitly collapsible navigation rail, integrate the top bar
with the workspace, and apply one consistent interaction hierarchy across
Scenarios, Configure, Candidates, Generate, Figures, and History.

The scientific services, routes, current-only persisted data model, candidate
selection semantics, artifact formats, and run contracts remain authoritative.
This is a presentation and UI-composition change plus one bounded local preview
service; it is not a frontend framework rewrite.

## Approved Direction

The approved visual direction is **Cartographic Workstation**:

- a deep teal instrument rail against a warm, lightly gridded canvas;
- real administrative boundaries and local terrain as the identifying visual
  language;
- restrained industrial typography and compact technical labels;
- generous card proportions and stable workspaces instead of narrow card walls;
- one emphasized decision action at a time;
- quiet motion that communicates state without decorating every interaction.

The interface remains offline-first. It does not download fonts, basemaps, or
decorative imagery. Local data makes the product distinctive.

## Product and Implementation Boundary

Retain NiceGUI, Quasar, Leaflet, Plotly, the existing page controllers, and the
current application routes. Do not introduce React, a separate SPA build, a
database, an online tile dependency, or a new catalog mutation workflow.

The implementation may:

- restructure NiceGUI page markup;
- extend shared presentation primitives;
- add CSS tokens and responsive layout classes;
- persist the navigation collapse preference in GUI settings;
- generate and serve validated local scenario preview files;
- add presentation-only models needed to keep page rendering testable.

The implementation must preserve unrelated uncommitted functional changes in
the current working tree. Existing callbacks and controller behavior are reused
unless a small presentation adapter is required.

## Visual System

Use the following core tokens as the basis of the revised stylesheet:

| Role | Value | Use |
| --- | --- | --- |
| Frame | `#082f31` | navigation and dark inspector surfaces |
| Ink | `#143536` | primary text |
| Muted ink | `#6d7d79` | descriptions and metadata |
| Canvas | `#efeee7` | page background |
| Surface | `#fbfaf6` | cards and controls |
| Accent | `#108879` | primary action and active state |
| Signal | `#b86f2d` | preparation and terrain-required state |
| Danger | `#b64747` | destructive action and blocking error |
| Border | `#d3dcd6` | surface separation |

Use local font stacks only. Headings use Bahnschrift SemiCondensed or a local
display fallback, body copy uses Aptos or a local UI-text fallback, and machine
values use Cascadia Code or `ui-monospace`. The visual identity comes primarily
from the cartographic content, proportions, and color system rather than a
network font.

Cards use 10-12 px radii, fine borders, and restrained shadows. Status badges
use semantic color and text together. Animations are limited to short hover
lifts, drawer transitions, skeleton fades, and active-job pulses; they are
disabled or reduced under `prefers-reduced-motion`.

## Shared Application Shell

### Navigation rail

The desktop rail has two explicit states:

- expanded: 224 px, with brand, icons, and labels;
- collapsed: 68 px, with icons and accessible tooltips.

The collapse control is pinned to the rail footer. The selected state is stored
as `navigation_collapsed` in the repository-local GUI settings. On viewports
below the existing desktop breakpoint, navigation becomes an overlay drawer and
the desktop collapse preference does not reduce the mobile touch target.

The settings document adopts one current shape with `language`, `output_roots`,
and the required boolean `navigation_collapsed`. A stale local settings document
without that field follows the existing current-only policy: load defaults and
write the current shape on the next preference update. Do not add a second
compatibility parser.

The rail owns the full viewport height. Its outer content uses `overflow-x:
hidden`, `min-width: 0`, `max-width: 100%`, and border-box sizing. Only the nav
item region may scroll vertically on unusually short screens. No horizontal
scrollbar is permitted in either rail state.

### Utility bar and page canvas

Replace the visually separate 72 px header with a 58 px workspace utility bar.
It sits only over the content region and shares the warm surface palette. It
contains:

- the current breadcrumb or workflow context;
- the active-job indicator;
- language selection;
- a compact toolkit identity control.

The utility bar does not repeat the page title. Page title, eyebrow, description,
and optional summary metrics live together in the page canvas. The main content
region is the only normal page scrollbar.

## Shared Action and Detail Primitives

Extend the existing presentation layer so pages use semantic action groups
instead of hand-built button rows.

The action hierarchy is:

1. **Primary**: one filled accent action for the current decision point.
2. **Secondary**: at most one adjacent outlined action when it is necessary to
   complete the same task.
3. **Tertiary**: ghost, icon, or overflow-menu actions for navigation and
   low-frequency operations.
4. **Danger**: separated text or menu actions that always retain their existing
   confirmation dialog.

Add shared primitives for an overflow menu, a right-side detail inspector, a
sticky action dock, a compact workflow step indicator, and consistent loading,
empty, warning, and error surfaces. Existing markers remain stable where tests
or page callbacks depend on them; new wrapper markers may be added.

Technical payloads, CLI commands, validation diagnostics, run details, and
profile management belong in an inspector or menu. They must not expand a grid
card to a different height.

## Scenario Catalog

### Page composition

Move the Registered, Ready, and Needs Preparation counts into the page header
and remove the separate summary strip. The catalog grid uses broad cards:

- content width of at least 1260 px: a six-track grid where ordinary cards span
  two tracks; if the final row has two cards, both span three tracks;
- content width from 720 through 1259 px: two equal columns;
- content width below 720 px: one column.

The current five-scenario catalog therefore forms a deliberate three-plus-two
composition instead of four narrow cards followed by one isolated card.

The general CLI guidance becomes one quiet full-width help strip after the grid.
It does not participate in the scenario-card layout.

### Card composition

Every card has a stable structure:

1. a wide map preview with an overlaid readiness badge and data-kind label;
2. scenario name and a single overflow control;
3. one short readiness description;
4. compact Boundary, DEM, and Profile availability indicators;
5. one full-width state-specific primary action.

Ready cards use **Configure and scan**. Preparation cards use **View setup
guidance**. Quick validation and full checksum run from the overflow menu and
show their progress and results in the right-side inspector. Dataset details and
copyable CLI commands open that same inspector. Existing validation callbacks
and background-job behavior remain unchanged.

### Local preview rules

The preview must represent registered local data honestly:

- with a usable DEM: render bounded elevation color plus hillshade, mask the
  raster to the administrative geometry, and outline the boundary;
- without a usable DEM: render the administrative boundary on a quiet neutral
  grid, without simulated terrain or an online basemap;
- with an unreadable boundary: render a neutral fallback cover and expose a
  concise preview diagnostic without disabling the scenario's existing action.

Preview failures are presentation failures, not catalog-readiness decisions.
The catalog's existing readiness model remains the source of truth.

## Scenario Preview Service

Add a framework-light `ScenarioPreviewService` outside the page renderer. It
accepts immutable requests containing validated repository-local boundary and
optional DEM paths plus their fingerprints. It returns an immutable result with
the preview path, preview kind (`terrain`, `boundary`, or `fallback`), and a
bounded diagnostic when degradation occurred.

The service:

- reads the boundary in its declared CRS and transforms it to EPSG:3857;
- computes a padded extent for a fixed 760 x 360 catalog cover;
- reuses the existing bounded DEM rendering behavior and combines it with the
  boundary outline;
- writes a bounded RGB preview through an atomic temporary-file replacement;
- stores results under `.lte-data/cache/scenario-previews/`;
- never serves a source shapefile or GeoTIFF directly.

The cache key includes a preview-style version, output dimensions, boundary
fingerprint, DEM fingerprint when present, and rendering parameters. A changed
manifest/file fingerprint or style version creates a new cache entry. Cached
files are validated as ordinary local images before reuse.

Uncached preview work runs through `nicegui.run.cpu_bound` using a top-level
worker and picklable, immutable requests. The page initially shows a neutral
skeleton and replaces it with the final terrain, boundary-only, or fallback
result only while the originating client is alive. Preview work does not occupy
the application-wide scientific `JobCoordinator` and does not appear as a
user-started job in the utility bar.

Register each validated output with a narrow static-file route, following the
existing per-file map and figure asset pattern. Do not expose the entire cache
directory.

## Workflow Pages

### Configure

Add a compact Configure -> Select -> Generate step indicator. Put profile
selection and saved/dirty state in a short toolbar. Copy, rename, set-default,
and delete move into the profile overflow menu; delete retains its confirmation.

Use a two-column work area: grouped form sections on the left and a sticky run
summary on the right. The bottom action dock contains Discard as tertiary, Save
draft as secondary, and Start scan as the only primary action. On narrow screens
the summary follows the form and the action dock becomes a safe full-width stack.

### Candidate Explorer

Keep the map dominant and use the existing candidate inspector as the stable
right column. Layer toggles remain compact controls on the map. Candidate
previous/next navigation and direct-number selection become one inspector
control rather than four competing buttons.

Scan actions are state-dependent: idle exposes Start scan; running exposes
progress and Cancel; Force rescan moves to the overflow menu after a result is
available. Confirm candidate is the only primary decision action and sits in a
map-bottom confirmation dock. Filmstrip remains available as an alternate view
without duplicating primary actions.

### Generate

Keep the artifact checklist and selection summary, but present them as one
publication workspace. Generate artifacts is the only primary action before a
run. Open in Figures becomes the next primary action only after generation has
produced an eligible source. Per-artifact status stays aligned with its artifact
row.

### Figures

Use a compact source bar followed by a large preview and right-side style
inspector. Load is secondary; Current selection moves to the source overflow
when it is available. Refresh preview is a local preview action, not a page-level
primary action.

Formats and destination state live in the bottom export dock. Export figures is
the only primary publication action. Existing source validation, bounded preview,
CPU-bound export, and technical diagnostics remain unchanged.

### History

Replace the vertical card-and-button wall with a scan-friendly run ledger. Each
row shows run identity, relationship, local time, status, and a concise artifact
summary. Derived runs are visually indented beneath or labeled relative to their
parent.

A row exposes one state-specific action: Open in Figures when eligible, otherwise
Inspect. Inspect, retry missing artifacts, reveal directory, and other
low-frequency actions live in the row overflow menu. Pending in-memory
selections remain a visually distinct section above published runs; Continue
generation is their primary action and Open in Figures moves to their menu.

## State, Error, and Empty Behavior

- Keep field validation next to the affected control.
- Keep task errors next to the task surface that produced them.
- Put raw exceptions, machine codes, paths, and copyable diagnostics in the
  detail inspector.
- When an action starts, disable only conflicting controls and change the action
  surface to an explicit running state.
- Do not leave stale previews or successful-looking primary actions visible after
  a source or form becomes invalid.
- Empty pages retain one clear recovery action and short explanatory copy.
- A scenario-cover failure never prevents validation, configuration, or setup
  guidance.

## Responsive and Accessibility Requirements

Desktop composition is the primary target, but all existing mobile accessibility
requirements remain in force:

- touch targets remain at least 44 CSS pixels on narrow/touch layouts;
- two-column workspaces collapse to one column without horizontal overflow;
- sticky docks account for safe-area insets and do not cover focused controls;
- collapsed navigation icons have accessible names and visible tooltips;
- menus, drawers, segmented controls, and dialogs expose keyboard focus and
  appropriate ARIA roles;
- status is never communicated by color alone;
- visible focus indicators meet the existing contrast direction;
- map and image previews retain meaningful alternative text or accessible
  summaries.

## Testing and Verification

Add focused tests before implementation behavior for:

- preview request validation and repository-path containment;
- real fixture boundary rendering;
- DEM, boundary-only, and fallback preview kinds;
- deterministic cache keys, cache reuse, and fingerprint/style invalidation;
- atomic preview writes and rejection of malformed cached images;
- navigation-collapse settings validation and persistence;
- one visible primary action per relevant page state;
- overflow menus retaining every existing callback and marker contract;
- responsive shell classes, no rail horizontal overflow, and mobile touch
  targets;
- preview failures leaving scenario workflow actions enabled;
- state-driven Candidate, Generate, Figures, and History action transitions.

Run focused GUI and preview-service tests during development, then run Ruff,
Python bytecode compilation, the complete Pytest suite, and `git diff --check`.
Finally start the local GUI and visually inspect every route at desktop width,
collapsed-rail width, tablet width, and narrow mobile width. Verify both English
and Simplified Chinese, reduced motion, keyboard focus, and real Chicago/New York
terrain previews plus boundary-only scenarios.

## Acceptance Criteria

The redesign is complete when:

- the navigation rail has no horizontal scrollbar and can be explicitly
  collapsed and restored;
- the utility bar and page canvas read as one application shell;
- Scenario Catalog uses real cached local previews and the approved broad-card
  composition;
- every scenario card exposes one visible primary action and an overflow menu;
- Configure, Candidates, Generate, Figures, and History match the approved
  workspace compositions and action hierarchy;
- no scientific workflow, route, current-only data contract, or offline boundary
  is weakened;
- focused and full verification pass; and
- visual inspection shows no clipped controls, accidental button wrapping, or
  horizontal page overflow at the supported widths.
