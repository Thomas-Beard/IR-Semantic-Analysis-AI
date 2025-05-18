# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['IR_Main2.py'],
    pathex=[],
    binaries=[],
    datas=[('cran.all.1400.xml', '.'), ('cran.qry.xml', '.'), ('cranqrel.trec.txt', '.'), ('collection_updates.py', '.'), ('..\\\\temp.bat', '.'), ('..\\\\zookeeper', 'zookeeper'), ('..\\\\solr-9.5.0', 'solr-9.5.0'), ('..\\\\jdk-17', 'jdk-17')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide6', 'shiboken6', 'QT-PyQt-PySide-Custom-Widgets'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
