# CodeMap installer — release + hosting runbook (2026-09-03)

## The artifact — ONE file (owner decision 03.09)

| file | size | sha256 |
|---|---|---|
| `codemap-setup-1.1.0.exe` | 22.4 MB | `45e9afc45324bfaa6b1d32c6ec8af29ba145f49209549c4bbc89f5ab00be48cb` |

Single-file web installer (the VS Code bootstrapper pattern): setup.exe
carries the app + graph pack + embedded Python 3.12 + llama.cpp CPU; the
2.5 GB navigator model is DOWNLOADED by the wizard from our OVH container
with SHA-256 verification (`DownloadTemporaryFile`). Bundling the model was
rejected on facts: Inno's single-exe ceiling is ~2.1 GB compressed and
requantizing the frozen r2.2 model to fit would break FREEZE-v1.

- OFFLINE path: a `codemap-lora-r22-q4_k_m.gguf` placed next to setup.exe
  is copied instead of downloaded (side-by-side, kept from v1).
- Updates/repairs skip the download when `{app}\bin\models\` already has
  the file (`onlyifdoesntexist` + `NeedModelDownload`).
- Silent installs work: Inno drives `NextButtonClick` at `wpReady` itself,
  so `/VERYSILENT` still downloads (or picks up the side-by-side gguf).

E2E matrix (v1.0 3-file build, 03.09): fresh interactive install (all wizard
pages) → boot on the embedded Python (`ladybug: true`, 1,415 entities, model
detected) → 6-second upgrade-in-place (model skipped) → uninstall with the
data-question (both keep and wipe paths) → zero residue, registry clean.
The one bug E2E caught is fixed at source: embedded Python's ._pth does not
add the script dir to sys.path — server.py now pins it itself.
Single-file build re-verified silent: `/VERYSILENT` fresh install downloads
the model from OVH (SHA-256 pass) and boots.

UNSIGNED — Certum signing is the owner's manual step (same as geodoc;
SignTool line ready in codemap.iss).

## Hosting — LIVE on OVH Object Storage (2026-09-03)

Created via the OVH API (application `checkItOut`, credentials in
`~/.ovh.conf` — NEVER in a repo): Public Cloud project `CheckItOutProd`,
Swift container **`downloads`**, region **WAW**, `containerType: public`.
Artifacts live under the `codemap/` prefix:

```
https://storage.waw.cloud.ovh.net/v1/AUTH_62ce8c0b4d874faa89fb3e086832f1a6/downloads/codemap/
  codemap-setup-1.1.0.exe          <- the ONE download users need
  codemap-lora-r22-q4_k_m.gguf     <- fetched BY the installer (SHA-256 pinned in codemap.iss)
  SHA256SUMS.txt
```

The user downloads the exe and runs it. Re-releasing = rebuild, then re-run
the uploader (scratchpad `swift_upload.py` pattern: OVH API
`POST /cloud/project/<sn>/storage/access` for a token, streamed Swift PUTs).
A model change means: new gguf upload + new `Gguf4BSha256` in codemap.iss +
rebuild — the hash pin makes a stale-cache download fail loudly, never
silently. A vanity URL (`downloads.check-it-out.pl`) stays possible later via
Cloudflare in front of the container; the storage URL is canonical for now.

Fallback boxes, for the record (state 03.09): vpsnew 51.38.135.102 = FIDO2
YubiKey touch, nginx config immutable — file-drop only; softmax 51.83.248.52
REBUILT (host key rotated to ED25519 `SHA256:B+6Doh…`, `id_ed25519_softmax`
no longer authorized); `ovh-server` ssh config points at a missing
`~/.ssh/ovhPrivate`.

## Rebuild

`powershell -ExecutionPolicy Bypass -File installer\build_installer.ps1`
(payload prep is idempotent; Inno compile ~12 s; bump `AppVersion` in
codemap.iss for a new release — AppId NEVER changes).
