@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation.
REM Uses "python -m sphinx" so the project's interpreter is used.
REM Override with: set SPHINXBUILD=sphinx-build

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=python -m sphinx
)
set SOURCEDIR=source
set BUILDDIR=build

if "%1" == "" goto help

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
