unit kgh_tools;

{$MODE Delphi}

interface

uses Windows, SysUtils;

const
  BELOW_NORMAL_PRIORITY_CLASS     = $00004000;

type
  TExecuteWaitEvent = procedure(const ProcessInfo: TProcessInformation; var ATerminate: Boolean) of object;

function ExecuteFileExternalWait(const AFilename: String; AParameter, ACurrentDir: String; AWait: Boolean; Show: Boolean; AOnWaitProc: TExecuteWaitEvent=nil) : TProcessInformation;
function ExecuteFile(const AFilename: String; AParameter, ACurrentDir: String; AWait: Boolean; Show: Boolean; AOnWaitProc: TExecuteWaitEvent=nil) : TProcessInformation;

implementation

function ExecuteFileExternalWait(const AFilename: String; AParameter, ACurrentDir: String; AWait: Boolean; Show: Boolean; AOnWaitProc: TExecuteWaitEvent=nil) : TProcessInformation;
var
  si: TStartupInfo;
begin
  if Length(ACurrentDir) = 0 then ACurrentDir:=ExtractFilePath(AFilename);
  if AnsiLastChar(ACurrentDir) = '\' then Delete(ACurrentDir, Length(ACurrentDir), 1);
  FillChar(si, SizeOf(si), 0);
  with si do
    begin
      cb:=SizeOf(si);
      dwFlags:=STARTF_USESHOWWINDOW;
      if show then wShowWindow:=SW_NORMAL
       else wShowWindow:=SW_HIDE;
    end;
  FillChar(result, SizeOf(result), 0);
  AParameter:=Format('"%s" %s', [AFilename, TrimRight(AParameter)]);
  CreateProcess(nil, PChar(AParameter), Nil, Nil, False,
                   CREATE_DEFAULT_ERROR_MODE or CREATE_NEW_CONSOLE or
                   BELOW_NORMAL_PRIORITY_CLASS, Nil, PChar(ACurrentDir), si, result)
end;

function ExecuteFile(const AFilename: String; AParameter, ACurrentDir: String; AWait: Boolean; Show: Boolean; AOnWaitProc: TExecuteWaitEvent=nil) : TProcessInformation;
var
  si: TStartupInfo;
  bTerminate: Boolean;
begin
  bTerminate:=False;
  if Length(ACurrentDir) = 0 then ACurrentDir:=ExtractFilePath(AFilename);
  if AnsiLastChar(ACurrentDir) = '\' then Delete(ACurrentDir, Length(ACurrentDir), 1);
  FillChar(si, SizeOf(si), 0);
  with si do
    begin
      cb:=SizeOf(si);
      dwFlags:=STARTF_USESHOWWINDOW;
      if show then wShowWindow:=SW_NORMAL
       else wShowWindow:=SW_HIDE;
    end;
  FillChar(result, SizeOf(result), 0);
  AParameter:=Format('"%s" %s', [AFilename, TrimRight(AParameter)]);
  if CreateProcess(nil, PChar(AParameter), Nil, Nil, False,
                   CREATE_DEFAULT_ERROR_MODE or CREATE_NEW_CONSOLE or
                   BELOW_NORMAL_PRIORITY_CLASS, Nil, PChar(ACurrentDir), si, result) then
    try
      if AWait then
        while WaitForSingleObject(result.hProcess, 1) <> Wait_Object_0 do
        begin
          if Assigned(AOnWaitProc) then
            begin
              AOnWaitProc(result, bTerminate);
              if bTerminate then TerminateProcess(result.hProcess, Cardinal(-1));
            end;
        end;
    finally
      FileClose(result.hProcess); { *Konvertiert von CloseHandle* }
      FileClose(result.hThread); { *Konvertiert von CloseHandle* }
    end;
end;

end.
