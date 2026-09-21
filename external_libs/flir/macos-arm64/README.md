# Copied FLIR runtime for the standalone Mac application

`runtime-4.4.0.246-cp312.tar.gz` contains the tested Spinnaker 4.4.0.246
runtime, matching Python 3.12 ARM64 bindings, USB GenTL transport and native
runtime dependencies. It is pinned to Apple Silicon/macOS 26+. `SHA256SUMS`
verifies the build input. These are runtime files, not a full SDK installer.

The archive extracts as `spinnaker/` with `lib/`, `python/` and `licenses/`.
`tools/bundle_flir.py` copies the libraries into the app, rewrites their links
to package-relative locations and signs the resulting bundle. FLIR dependencies
are isolated from OpenCV's libraries to prevent incompatible version collisions.

Vendor and third-party libraries retain their own licenses, supplied in the
archive and copied into the app. They are not relicensed under this repository's
MIT license. The FLIR component is intended for the supported FLIR camera backend.

The runtime was assembled from the user's Spinnaker 4.4.0.246 Mac installer and
the dependency versions used to validate the application. Updating it is a deliberate
packaging change; the workflow does not download a newer SDK automatically.
