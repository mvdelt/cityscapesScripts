#!/usr/bin/env python
#
# Enable cython support for slightly faster eval scripts:
# python -m pip install cython numpy
# CYTHONIZE_EVAL= python setup.py build_ext --inplace
#
# For MacOS X you may have to export the numpy headers in CFLAGS
# export CFLAGS="-I /usr/local/lib/python3.6/site-packages/numpy/core/include $CFLAGS"

import os
from setuptools import setup, find_packages

include_dirs = []
ext_modules = []
if 'CYTHONIZE_EVAL' in os.environ:
    from Cython.Build import cythonize
    import numpy as np
    include_dirs = [np.get_include()]

    os.environ["CC"] = "g++"
    os.environ["CXX"] = "g++"

    pyxFile = os.path.join("cityscapesscripts",
                           "evaluation", "addToConfusionMatrix.pyx")
    ext_modules = cythonize(pyxFile)

# i.21.3.7.13:31) 에러 해결 위해 encoding='utf-8'로 지정해줬음. ->예상대로 이제 에러없이 잘 됨.
#  요약: 인코딩을 지정해주지 않으면, 한글 윈도 기본인코딩인 cp949로 읽으려하게되는데 cp949로 읽을수없는게 있어서 에러뜨는것. 
#   참고로, 그냥 vscode로 열어보면 자동으로 utf-8로 읽어서 잘 보이는거라함.
#  에러: UnicodeDecodeError: 'cp949' codec can't decode byte 0xe2 in position 304: illegal multibyte sequence
#  해결책 참조: https://daewonyoon.tistory.com/296
with open("README.md", encoding='utf-8') as f:
    readme = f.read()

with open(os.path.join('cityscapesscripts', 'VERSION')) as f:
    version = f.read().strip()

console_scripts = [
    'csEvalPixelLevelSemanticLabeling = cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling:main',
    'csEvalInstanceLevelSemanticLabeling = cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling:main',
    'csEvalPanopticSemanticLabeling = cityscapesscripts.evaluation.evalPanopticSemanticLabeling:main',
    'csEvalObjectDetection3d = cityscapesscripts.evaluation.evalObjectDetection3d:main',
    'csCreateTrainIdLabelImgs = cityscapesscripts.preparation.createTrainIdLabelImgs:main',
    'csCreateTrainIdInstanceImgs = cityscapesscripts.preparation.createTrainIdInstanceImgs:main',
    'csCreatePanopticImgs = cityscapesscripts.preparation.createPanopticImgs:main',
    'csDownload = cityscapesscripts.download.downloader:main',
    'csPlot3dDetectionResults = cityscapesscripts.evaluation.plot3dResults:main'
]

gui_scripts = [
    'csViewer = cityscapesscripts.viewer.cityscapesViewer:main [gui]',
    'csLabelTool = cityscapesscripts.annotation.cityscapesLabelTool:main [gui]'
]

config = {
    'name': 'cityscapesScripts',
    'description': 'Scripts for the Cityscapes Dataset',
    'long_description': readme,
    'long_description_content_type': "text/markdown",
    'author': 'Marius Cordts',
    'url': 'https://github.com/mcordts/cityscapesScripts',
    'author_email': 'mail@cityscapes-dataset.net',
    'license': 'https://github.com/mcordts/cityscapesScripts/blob/master/license.txt',
    'version': version,
    'install_requires': ['numpy', 'matplotlib', 'pillow', 'appdirs', 'pyquaternion', 'coloredlogs', 'tqdm', 'typing'],
    'setup_requires': ['setuptools>=18.0'],
    'extras_require': {
        'gui': ['PyQt5']
    },
    'packages': find_packages(),
    'scripts': [],
    'entry_points': {'gui_scripts': gui_scripts,
                     'console_scripts': console_scripts},
    'package_data': {'': ['VERSION', 'icons/*.png']},
    'ext_modules': ext_modules,
    'include_dirs': include_dirs
}

setup(**config)
