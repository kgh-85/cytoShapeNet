unit main;

{$MODE Delphi}

interface

uses
  Dialogs, Forms, SysUtils, Variants, Classes, Windows, StdCtrls, Spin,
  ExtCtrls, Menus, kgh_tools;

type

  { TFrmMain }

  TFrmMain = class(TForm)
    btnProcess: TButton;
    cbxKeepOBJ: TCheckBox;
    cbxIncludeSubdirs: TCheckBox;
    lbIntro: TLabel;
    MainMenu1: TMainMenu;
    Help: TMenuItem;
    SelectDirectoryDialog: TSelectDirectoryDialog;
    seThreadCount: TSpinEdit;
    lbThreadcount: TLabel;
    gpFiji: TGroupBox;
    lbThreshold: TLabel;
    edThreshold: TEdit;
    gpGeneral: TGroupBox;
    tmStart: TTimer;
    procedure btnProcessClick(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure FormCloseQuery(Sender: TObject);
    procedure HelpClick(Sender: TObject);
    procedure tmStartTimer(Sender: TObject);
  private
    { Private-Deklarationen }
    procedure processFolder(folder: String);
  public
    { Public-Deklarationen }
  end;

var
  FrmMain: TFrmMain;

implementation

{$R *.lfm}

function ListFilesRecursive(Path, Mask: String; ShowPath: Boolean): TStringList;
var
  SR: TSearchRec;
  Erg, I: Integer;
  RelDir, TempStr : String;
  TempList: TStringList;
begin
  IncludeTrailingBackslash(Path);
  RelDir:=ExtractFilePath(Mask);
  Mask:=ExtractFileName(Mask);
  result:=TStringList.Create;
  result.Duplicates:=dupIgnore;
  result.Sorted:=True;
  TempList:=TStringList.Create;
  Erg:=FindFirst(Path+RelDir+'*.*', faDirectory, SR);
  while Erg=0 do
    begin
      if (SR.attr and faDirectory)<>0 then
       if SR.Name[1]<>'.' then
        begin
          TempList.Assign(ListFilesRecursive(Path, RelDir+SR.Name+'\'+Mask, ShowPath));
          for I:=0 to TempList.Count-1 do
            begin
              TempStr:=TempList[I];
              if ShowPath and not (TempStr[2]=':') then TempStr:=Path+TempStr;
              result.Add(ExtractFilePath(TempList[I]));
            end;
        end;
      Erg:=FindNext(SR);
    end;
  Erg:=FindFirst(Path+RelDir+Mask, $27, SR);
  while Erg = 0 do
    begin
      if not ShowPath then
      result.Add(RelDir)
      else
      result.Add(Path+RelDir);
      Erg:=FindNext(SR);
    end;
  SysUtils.FindClose(SR);
  TempList.Free;
end;


procedure TFrmMain.processFolder(folder: String);
var
  sl, subdirs: TStringList;
  targetFolder, baseFolder, baseFolderEscaped, outPath, path: String;
  i, j: Integer;
  SR: TSearchRec;
  pi: array of TProcessInformation;
  running, dataToConvert: boolean;
begin
  path:=ExtractFilePath(Application.Exename);
  sl:=TStringList.Create;
  // get all subdirectories
  subdirs:=ListFilesRecursive(folder + '\', '*.*', true);
    try
      for j:=0 to subdirs.Count-1 do
        begin
          //outPath:=subdirs[j] + '\';
          targetFolder:= StringReplace(subdirs[j], '\', '/', [rfReplaceAll]);

          if FindFirst(subdirs[j] + '*.tif', faAnyFile, SR) = 0 then
            begin
              repeat
                  sl.Add(subdirs[j] + SR.Name); //Fill the list
              until FindNext(SR) <> 0;
              SysUtils.FindClose(SR);
            end;
          if sl.Count = 0 then
            begin
              if FindFirst(subdirs[j] + '*.obj', faAnyFile, SR) = 0 then
                begin
                  repeat
                      sl.Add(subdirs[j] + SR.Name); //Fill the list
                  until FindNext(SR) <> 0;
                  SysUtils.FindClose(SR);
                end;
            end;

          for i:=0 to sl.Count-1 do
           if not (FileExists(sl[i] + '.dat') and FileExists(sl[i] + '.obj')) then break;
          dataToConvert:=not (i = sl.Count);

          if dataToConvert then
            begin
              // limit threads to imagecount
              if sl.Count < seThreadCount.Value then seThreadCount.Value:=sl.Count;
              setLength(pi, seThreadCount.Value);

              sl.LoadFromFile(path + 'ImageJ\BatchFolder_TIFF_to_OBJ_macro.ijm.template');
              baseFolder:=StringReplace(ExtractFilePath(Application.ExeName), '\', '/', [rfReplaceAll]);
              baseFolderEscaped:=baseFolder;
              insert('\"', baseFolderEscaped, 4);
              baseFolderEscaped:=baseFolderEscaped + '\"';
              for i:=0 to sl.Count-1 do
                if pos('threshold', sl[i]) > 0 then sl[i]:=StringReplace(sl[i], 'threshold=50', 'threshold=' + edThreshold.Text, []);
              sl.Insert(0, 'outputFolder = "' + targetFolder + '";' + #10#13);
              sl.Insert(0, 'inputFolder = "' + targetFolder + '";');
              sl.Insert(0, 'baseFolder = "' + baseFolder + '";');
              sl.Insert(0, 'baseFolderEscaped = "' + baseFolderEscaped + '";');

              sl.SaveToFile(path + 'ImageJ\BatchFolder_TIFF_to_OBJ_macro.ijm');
              // Multithread start
              for i:=1 to seThreadCount.Value do
                begin
                  sl.LoadFromFile(path + 'ImageJ\batch_convert.bat.template');
                  sl[0]:=sl[0] + ' "' + IntToStr(i) + ' ' + IntToStr(seThreadCount.Value) + '"';
                  sl.SaveToFile(path + 'ImageJ\batch_convert_' + IntToStr(i) + '.bat');
                  pi[i-1]:=ExecuteFileExternalWait(path + 'ImageJ\batch_convert_' + IntToStr(i) + '.bat', path, '', false, true);
                end;

              running:=true;
              while running do
                begin
                  // Wait for multithreaded processes to finish
                  Application.ProcessMessages;
                  for i:=0 to High(pi) do
                   if WaitForSingleObject(pi[i].hProcess, 10) > 0 then break;
                  if i = High(pi) then running:=false;
                end;

              for i:=0 to High(pi) do
                begin
                  FileClose(pi[i].hProcess);
                  FileClose(pi[i].hThread);
                end;

              outPath:=ExtractFilePath(folder + '\');
              // Multithread stop
              if not cbxKeepOBJ.Checked then
                begin
                  if FindFirst(outPath + '*.obj', faAnyFile, SR) = 0 then
                    begin
                      repeat
                          sl.Add(outPath + SR.Name); //Fill the list
                      until FindNext(SR) <> 0;
                      SysUtils.FindClose(SR);
                    end;
                  for i := 0 to sl.Count-1 do DeleteFile(PCHAR(sl[i]));
                end;
            end;
        end;
    finally
      sl.Free;
      subdirs.Free;
    end;
  MessageDlg('Information', 'File conversion completed.', mtInformation, [mbOK], 0);
  Close;
end;

procedure TFrmMain.btnProcessClick(Sender: TObject);
begin
  SelectDirectoryDialog.InitialDir:=ExtractFilePath(Application.ExeName);
  if not SelectDirectoryDialog.Execute then exit;
  ProcessFolder(SelectDirectoryDialog.FileName);
end;

procedure TFrmMain.FormCloseQuery(Sender: TObject);
begin
  halt;
end;

procedure TFrmMain.HelpClick(Sender: TObject);
const
  br = #10#13;
begin
  MessageDlg('Information', 'Commandline parameters:' + br
    + br
    + '-h or -? or --help or --info' + br + 'Display help' + br
    + br
    + '--threadCount=X' + br + 'Define "X" as Threadcount (defaults to current CPU threads' + br
    + br
    + '-k or --keepObj' + br + 'Keep the OBJ file in the conversion process' + br
    + br
    + '-e or --excludeSubdirs' + br + 'Skip subdirectories of [folder]' + br
    + br
    + '--objThreshhold=XXX' + br + 'Define "XXX" as the OBJ isosurface threshold' + br
    + br
    + '--folder' + br + 'Folder to process'
  , mtInformation, [mbOK], 0);
end;

procedure TFrmMain.tmStartTimer(Sender: TObject);
var
  folder: string;
begin
  tmStart.enabled:=False;
  if (Application.HasOption('h', 'help') OR Application.HasOption('?', 'info'))
   then HelpClick(Sender);

  if Application.HasOption('t', 'threadCount')
   then seThreadCount.Value:= StrToInt(Application.GetOptionValue('t', 'threadCount'));

  if Application.HasOption('k', 'keepObj')
   then cbxKeepOBJ.checked:=True;

  if Application.HasOption('e', 'excludeSubdirs')
   then cbxIncludeSubdirs.checked:=False;

  if Application.HasOption('o', 'objThreshhold')
   then edThreshold.Text:=Application.GetOptionValue('o', 'objThreshhold');

  if Application.HasOption('f', 'folder')
   then
    begin
      folder:=Application.GetOptionValue('f', 'folder');
      // check for relative path param
      if strpos(':', PChar(folder)) = nil
       then folder:=ExtractFilePath(Application.Exename) + folder;
      ProcessFolder(folder);
    end;
end;

procedure TFrmMain.FormCreate(Sender: TObject);
begin
  seThreadCount.Value:=System.CPUCount;
end;

end.
