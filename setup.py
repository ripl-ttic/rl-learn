from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))


setup(name='rl_learn',
      packages=[package for package in find_packages()
                if package.startswith('rl_learn')],
      install_requires=[
          'numpy',
          'matplotlib',
          'imageio',
          'pandas',
          'gin-config'
          # 'pytorch',
          # 'torchvision',
          # 'mpi4py',
      ],
      description='PyTorch implementation of RL-LEARN',
      author='Bobby Shi',
      author_email='bhshi@uchicago.edu',
version='0.0')