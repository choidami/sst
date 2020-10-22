from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="kruskals_cpp",
      ext_modules=[cpp_extension.CppExtension("kruskals_cpp", ["kruskals.cpp"])],
      cmdclass={"build_ext": cpp_extension.BuildExtension})