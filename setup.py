import sys
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
    "packages": ["os", "tkinter", "threading"],
    "excludes": [],
    "include_files": []
}

# base="Win32GUI" should be used only for Windows GUI app
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="MyAutoResxTranslator",
    version="1.0",
    description="A GUI for translating .resx files with DeepL",
    options={"build_exe": build_exe_options},
    executables=[Executable("app.py", base=base, target_name="MyAutoResxTranslator.exe")]
)
