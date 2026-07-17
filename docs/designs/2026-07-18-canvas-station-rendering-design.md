# Canvas Station Rendering Design

## Problem

The Chicago candidate page renders 2,168 stations as default Leaflet pin
markers. Those markers create a large DOM surface, overlap heavily at the city
extent, and sit above vector paths in Leaflet's marker pane. During a real fast
scan, 23,192,505 grid positions produce 5,355 progress events over roughly 13
seconds. The combination makes the embedded browser lag and eventually become
unresponsive while candidate rectangles remain difficult to see.

The same real DEM, boundary, and station payload renders reliably when scanning
is disabled, so the source data and DEM overlay are not the failure boundary.

## Goals

- Preserve every station as an individual point.
- Preserve click access to the existing station metadata popup.
- Render stations without one DOM marker per feature.
- Keep candidate rectangles visually above stations throughout scanning.
- Reduce scan-driven browser redraw pressure without dropping candidate deltas
  or final scan state.
- Apply the same rendering behavior to the GUI and standalone web selector.

## Non-goals

- Do not cluster, aggregate, sample, or hide stations.
- Do not change candidate validity, ordering, caching, or scan computation.
- Do not add an external map or clustering dependency.
- Do not change the optional online basemap behavior.

## Design

### Station layer

Add a small local Leaflet extension that registers a `stationDots` layer
factory. The factory wraps `L.geoJSON` and supplies `pointToLayer`, returning an
interactive `L.circleMarker` for every station. The candidate map already uses
`preferCanvas: true`, so the circles share Leaflet's canvas renderer instead of
creating thousands of pin-marker DOM elements.

Dots use a fixed screen-space radius of 2.5 pixels, a restrained blue fill, and
moderate transparency. They remain individually clickable. The existing
`eachLayer` popup binding continues to attach escaped station metadata to every
circle.

The extension is packaged with the GUI assets, served locally, and loaded
through the Leaflet element's `additional_resources` option. No network access
or third-party package is introduced.

### Candidate visibility

Candidate rectangles stay in Leaflet's `overlayPane`, sharing the same canvas
renderer as station dots. Stations are added first and candidate rectangles are
added afterward, so Leaflet draws the rectangles above the dots while retaining
one interactive canvas. A separate higher pane is unsuitable because its
full-size canvas intercepts clicks intended for station dots below it. Existing
selected/unselected colors and fill behavior remain unchanged.

### Progressive updates

The page continues to consume every scanner progress event and every candidate
addition or removal. Its UI drain timer changes from 150 ms to 250 ms, limiting
browser-facing refreshes to four per second while preserving the authoritative
final result and cancellation behavior.

### Shared integration

The candidate page receives the local station-layer resource URL explicitly.
Both the full GUI application and the standalone web selector register and pass
the same packaged resource. Framework-light controller and state code remains
unchanged.

## Failure handling

The station-layer resource is local and versioned with the package. If it cannot
be loaded, the browser reports a resource error rather than silently falling
back to DOM pins. Existing candidate-session and map-preparation error handling
is unchanged.

## Verification

- Unit-test that station data uses the `stationDots` factory with interactive
  circle styling and no clustering.
- Unit-test that candidate rectangles request the shared overlay pane.
- Verify package metadata includes the JavaScript resource.
- Run the complete Python test suite, Ruff, compile checks, and `git diff
  --check`.
- Run a real Chicago scan in the embedded browser and verify page identity,
  responsiveness during scanning, console health, clickable station dots, and
  visible candidate rectangles.
