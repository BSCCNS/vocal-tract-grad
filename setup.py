from setuptools import setup
import sys

sys.setrecursionlimit(5000)

APP = ["ngui.py"]
DATA_FILES = [
    ('', ['gfmdriver.py', 'gfm.py']),
]

OPTIONS = {
    'argv_emulation': True,
    'packages': [
        'librosa',
        'numpy',
        'scipy',
        'sounddevice',
        'nicegui',
        'soundfile',
    ],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
