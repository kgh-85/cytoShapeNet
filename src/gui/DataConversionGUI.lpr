program DataConversionGUI;

{$MODE Delphi}

uses
  Interfaces,
  Forms,
  main in 'main.pas' {Form1};

{.$R *.res}

begin
  Application.Initialize;
  Application.CreateForm(TFrmMain, FrmMain);
  Application.Run;
end.

