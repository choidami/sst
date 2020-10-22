from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='edmonds_cpp',
      ext_modules=[cpp_extension.CppExtension('edmonds_cpp', ['chuliu_edmonds.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})