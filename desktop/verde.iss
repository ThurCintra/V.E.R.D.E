; Instalador do V.E.R.D.E. (Inno Setup 6). Empacota a pasta dist\VERDE gerada pelo PyInstaller.
;
;     iscc /DAppVersion=1.0.0 desktop\verde.iss
;
; Instala só para o usuário atual, em %LOCALAPPDATA%\Programs\VERDE: não pede administrador e funciona em
; computadores de escola ou de empresa sem permissão de instalar programas para todos.
; O resultado vai para dist\instalador\VERDE-Instalador.exe.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{DEBF8F39-3AC4-427C-92E7-AB3C623B9320}
AppName=V.E.R.D.E.
AppVersion={#AppVersion}
AppVerName=V.E.R.D.E. {#AppVersion}
AppPublisher=V.E.R.D.E.
AppComments=Detector de resíduos recicláveis
; Sem administrador: tudo na pasta do usuário ({autopf} vira %LOCALAPPDATA%\Programs com "lowest")
PrivilegesRequired=lowest
DefaultDirName={autopf}\VERDE
DisableProgramGroupPage=yes
DisableDirPage=auto
DisableReadyPage=yes
UsePreviousAppDir=yes
CloseApplications=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
WizardStyle=modern
SetupIconFile=verde.ico
UninstallDisplayIcon={app}\VERDE.exe
UninstallDisplayName=V.E.R.D.E.
OutputDir=..\dist\instalador
OutputBaseFilename=VERDE-Instalador
; O PyTorch ocupa ~1 GB: lzma2 sem modo sólido comprime bem e não exige memória demais para instalar
Compression=lzma2/max
SolidCompression=no

[Languages]
Name: "ptbr"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\VERDE\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\LEIA-ME-WINDOWS.txt"; DestDir: "{app}"; Flags: ignoreversion

[InstallDelete]
; Numa atualização, a pasta do programa é trocada inteira (arquivos de versões antigas não ficam para trás).
; Os dados do usuário ficam em %LOCALAPPDATA%\VERDE e não são tocados.
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{autoprograms}\V.E.R.D.E."; Filename: "{app}\VERDE.exe"; Comment: "Detector de resíduos recicláveis"
Name: "{autodesktop}\V.E.R.D.E."; Filename: "{app}\VERDE.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\VERDE.exe"; Description: "{cm:LaunchProgram,V.E.R.D.E.}"; Flags: nowait postinstall skipifsilent
