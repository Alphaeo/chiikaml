# Depuis Python 3.8, le PATH seul ne suffit plus pour que `import`
# trouve les DLL dont un module natif (.pyd) depend -- il faut les
# enregistrer explicitement via os.add_dll_directory(). Sans ca,
# `import chiikaml` echoue avec "DLL load failed" (message peu
# parlant : il ne dit pas QUELLE dll manque), meme si toutes les DLL
# necessaires (runtime MinGW : libstdc++-6.dll, libgcc_s_seh-1.dll)
# sont bien presentes sur le PATH.
import os

if os.name == "nt":
    mingw_bin = r"C:\msys64\ucrt64\bin"
    if os.path.isdir(mingw_bin):
        os.add_dll_directory(mingw_bin)
