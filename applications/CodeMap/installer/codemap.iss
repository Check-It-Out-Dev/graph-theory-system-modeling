; CodeMap — Inno Setup 6 definition, the geodoc pattern adapted (see
; ../../geodoc/installer/geodoc.iss for the researched laws it inherits:
; per-user default without UAC, onlyifdoesntexist so updates never rewrite
; the 2.5 GB model, process cleanup before uninstall, silent uninstall
; keeps user data without a hanging MsgBox).
;
; ONE FILE (owner decision 03.09): setup.exe carries the app + graph pack
; + embedded Python 3.12 (PSF-signed binaries, real-ladybug vendored — the
; ONLY non-stdlib dep) + llama.cpp b10155 CPU. The 2.5 GB navigator model
; is NOT bundled — Inno's single-exe ceiling is ~2.1 GB compressed, and
; requantizing the frozen model to fit would break FREEZE-v1. Instead the
; wizard DOWNLOADS it from our OVH object storage during install, with
; SHA-256 verification (DownloadTemporaryFile) — the VS Code bootstrapper
; pattern. Updates and repairs skip the download when the file exists.
; OFFLINE path: a codemap-lora-r22-q4_k_m.gguf placed NEXT TO setup.exe
; is copied instead of downloaded (GOG-style side-by-side, kept from v1).
;
; The big 80B tier stays OPTIONAL by design: drop a Qwen3-Next-80B gguf
; into {app}\bin\models\ and the wizard detects it (graph-native Cypher).
;
; Build:  installer\build_installer.ps1   (or: iscc installer\codemap.iss)
; Output: installer\Output\codemap-setup-<wersja>.exe  (single file)

#define AppName "CodeMap"
#define AppVersion "1.1.0"
#define AppPublisher "Check-It-Out-Dev"
#define Gguf4B "codemap-lora-r22-q4_k_m.gguf"
#define Gguf4BSha256 "9c454526d7d0d1b05c3988130f32d3c032c94832d68a2d619501f7240d6ccf26"
#define Gguf4BUrl "https://storage.waw.cloud.ovh.net/v1/AUTH_62ce8c0b4d874faa89fb3e086832f1a6/downloads/codemap/codemap-lora-r22-q4_k_m.gguf"

[Setup]
; STABLE update identity — never change (upgrade-in-place, one entry in
; Apps & features)
AppId={{7C0DE3A9-11B0-4A57-9E2F-30D1C0DE4A90}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL=https://github.com/Check-It-Out-Dev/graph-theory-system-modeling
AppSupportURL=https://checkitout.app/codemap
AppUpdatesURL=https://checkitout.app/codemap
AppCopyright=© 2026 {#AppPublisher}
; per-user without UAC (VS Code convention: %LOCALAPPDATA%\Programs);
; the dialog still allows a conscious all-users install
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DefaultDirName={autopf}\CodeMap
DefaultGroupName=CodeMap
DisableProgramGroupPage=yes
DisableDirPage=auto
DisableWelcomePage=no
SetupMutex=codemap-setup-mutex
CloseApplications=yes
RestartApplications=no
MinVersion=10.0.17763
ArchitecturesInstallIn64BitMode=x64compatible
OutputBaseFilename=codemap-setup-{#AppVersion}
OutputDir=Output
LicenseFile=LICENSE.txt
Compression=lzma2/ultra64
SolidCompression=yes
SetupLogging=yes
UninstallDisplayName={#AppName}
WizardStyle=modern
; setup.exe metadata (Properties -> Details); the default 0.0.0.0 reads
; as amateur work — set explicitly (geodoc research)
VersionInfoVersion={#AppVersion}.0
VersionInfoCompany={#AppPublisher}
VersionInfoDescription={#AppName} installer — precomputed codebase understanding, navigated locally
VersionInfoCopyright=© 2026 {#AppPublisher}
VersionInfoProductName={#AppName}
; Certum signing (owner's manual step, order: rcedit BEFORE sign):
; SignTool=certum $f

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; \
  GroupDescription: "Additional shortcuts:"

[Files]
; the repo surface the wizard expects — {app} mirrors a clone
Source: "..\codemap.py"; DestDir: "{app}"
Source: "..\.env.example"; DestDir: "{app}"
Source: "..\README.md"; DestDir: "{app}"
Source: "LICENSE.txt"; DestDir: "{app}"
Source: "CodeMap.cmd"; DestDir: "{app}"
Source: "..\app\*"; DestDir: "{app}\app"; Excludes: "__pycache__\*"
Source: "..\graph\pack\*"; DestDir: "{app}\graph\pack"; Excludes: "*.wal,*.lock"
; gold questions + invalidation sets (the cache tier + benchable install);
; api_answers.jsonl (a local log) and raw/ mining outputs stay out
Source: "..\eval\q\mfq_all.jsonl"; DestDir: "{app}\eval\q"
Source: "..\eval\q\mfq_gold.jsonl"; DestDir: "{app}\eval\q"
Source: "..\eval\q\INVALIDATED_2026-09-02.json"; DestDir: "{app}\eval\q"
Source: "..\eval\q\INVALIDATED_2026-09-02-curated.json"; DestDir: "{app}\eval\q"
; training/ modules are RUNTIME imports (loop_runner -> datagen; rung_cpu
; -> master_prompt_v1.txt); data/ stays out
Source: "..\training\*.py"; DestDir: "{app}\training"; Excludes: "__pycache__\*"
Source: "..\training\master_prompt_v0.txt"; DestDir: "{app}\training"
Source: "..\training\master_prompt_v1.txt"; DestDir: "{app}\training"
Source: "..\training\FREEZE-v1.md"; DestDir: "{app}\training"
Source: "..\training\PIPELINE.md"; DestDir: "{app}\training"
Source: "..\docs\*"; DestDir: "{app}\docs"; Flags: recursesubdirs
; python embeddable — PSF-signed binaries (zero AV false alarms, unlike
; PyInstaller — geodoc research #8164); real-ladybug vendored inside
Source: "python-embed\*"; DestDir: "{app}\python-embed"; Flags: recursesubdirs
; llama.cpp CPU lands at the exact path the wizard checks
Source: "llama-bin\*"; DestDir: "{app}\bin\llama\b10155"; Flags: recursesubdirs
; MODEL, two external sources — never bundled (see header). An app update
; must never rewrite the 2.5 GB: onlyifdoesntexist on both. Repair of a
; corrupted file: delete bin\models\<file>.gguf and rerun the installer.
; (1) OFFLINE: a gguf beside setup.exe wins over the download
Source: "{src}\{#Gguf4B}"; DestDir: "{app}\bin\models"; \
  Flags: external skipifsourcedoesntexist onlyifdoesntexist
; (2) DOWNLOADED: fetched into {tmp} by the wizard (SHA-256 verified)
Source: "{tmp}\{#Gguf4B}"; DestDir: "{app}\bin\models"; \
  Flags: external onlyifdoesntexist; Check: ModelWasDownloaded

[Icons]
Name: "{autoprograms}\CodeMap"; Filename: "{app}\CodeMap.cmd"; \
  WorkingDir: "{app}"; Comment: "Ask a 430k-LOC codebase, offline"
Name: "{autodesktop}\CodeMap"; Filename: "{app}\CodeMap.cmd"; \
  WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\CodeMap.cmd"; Description: "Launch CodeMap"; \
  Flags: postinstall nowait skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\app\__pycache__"
Type: filesandordirs; Name: "{app}\training\__pycache__"

[Code]
var
  DownloadPage: TDownloadWizardPage;
  ModelDownloaded: Boolean;

function ModelWasDownloaded(): Boolean;
begin
  Result := ModelDownloaded;
end;

function NeedModelDownload(): Boolean;
begin
  // skip when an install already carries the model (update/repair) or a
  // side-by-side gguf sits next to setup.exe (offline path)
  Result := not FileExists(ExpandConstant('{app}\bin\models\{#Gguf4B}'))
        and not FileExists(ExpandConstant('{src}\{#Gguf4B}'));
end;

procedure InitializeWizard;
begin
  DownloadPage := CreateDownloadPage(SetupMessage(msgWizardPreparing),
    'Downloading the navigator model (2.5 GB, SHA-256 verified) from OVH…', nil);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  // silent installs pass through here too — Inno "clicks" the pages itself
  if (CurPageID = wpReady) and NeedModelDownload() then
  begin
    DownloadPage.Clear;
    DownloadPage.Add('{#Gguf4BUrl}', '{#Gguf4B}', '{#Gguf4BSha256}');
    DownloadPage.Show;
    try
      try
        DownloadPage.Download;
        ModelDownloaded := True;
      except
        if DownloadPage.AbortedByUser then
          Log('Model download aborted by user')
        else
          SuppressibleMsgBox(AddPeriod(GetExceptionMessage), mbCriticalError,
            MB_OK, IDOK);
        Result := False;
      end;
    finally
      DownloadPage.Hide;
    end;
  end;
end;

procedure CleanupProcesses();
var
  R: Integer;
  Cmd: String;
begin
  // geodoc drill 03.09: a surviving python.exe/llama-server.exe under
  // {app} BLOCKS file deletion — end the app's own processes (known
  // names only; never unins000 itself)
  Cmd := '-NoProfile -Command "Get-CimInstance Win32_Process | ' +
         'Where-Object { $_.ExecutablePath -like ''' +
         ExpandConstant('{app}') + '\*'' -and $_.Name -in @(' +
         '''python.exe'',''pythonw.exe'',''llama-server.exe'') } | ' +
         'ForEach-Object { Stop-Process -Id $_.ProcessId -Force } "';
  Exec('powershell.exe', Cmd, '', SW_HIDE, ewWaitUntilTerminated, R);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
    CleanupProcesses();
  if CurUninstallStep = usUninstall then
    // default KEEP (button NO); deleting data is a conscious decision.
    // A [Code] MsgBox ignores /SUPPRESSMSGBOXES and would hang a silent
    // uninstall forever (geodoc E2E finding 03.09) — silent mode keeps
    // the data without asking.
    if (not UninstallSilent) and (MsgBox('Also remove downloaded models and your configuration?' + #13#10 + #13#10
              + 'This includes: bin\models (the navigator model and any '
              + '80B gguf you dropped in) and your .env API key file.'
              + #13#10 + 'This cannot be undone.',
              mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES) then
    begin
      DelTree(ExpandConstant('{app}\bin\models'), True, True, True);
      DeleteFile(ExpandConstant('{app}\.env'));
    end;
end;
