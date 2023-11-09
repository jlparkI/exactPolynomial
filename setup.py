"""Package setup file."""
import os
import platform
import subprocess
from setuptools import setup, find_packages
import numpy
from Cython.Distutils import build_ext
#Do NOT put this package import before the others.
#setup tools monkey patches distutils, so reversing
#the import order will (unbelievably enough) lead to an error.
#If your linter tells you to put this first, IGNORE YOUR LINTER.
from distutils.extension import Extension


def get_version(setup_fpath):
    """Retrieves the version number."""

    os.chdir(os.path.join(setup_fpath, "exactPolynomial"))
    with open("__init__.py", "r") as fhandle:
        version_line = [l for l in fhandle.readlines() if
                    l.startswith("__version__")]
        version = version_line[0].split("=")[1].strip().replace('"', "")
    os.chdir(setup_fpath)
    return version



def initial_checks():
    """Checks that there IS a CUDA path and figures out what it is.
    Also checks that we are running on Linux, if not, raise an error."""
    if "linux" not in platform.system().lower():
        raise ValueError("This package will only install on Linux (it uses pthreads). "
            "Development of a windows-compatible version may occur later.")

    setup_fpath = os.path.dirname(os.path.abspath(__file__))
    #No Cuda will be set to True if any steps in the cudaHadamardTransform
    #pipeline fail.
    NO_CUDA = False
    if "CUDA_PATH" in os.environ:
        CUDA_PATH = os.environ["CUDA_PATH"]
    else:
        CUDA_PATH = "/usr/local/cuda"
    print(f"Using cuda located at: {CUDA_PATH}")

    if not os.path.isdir(CUDA_PATH):
        print("CUDA is not at the expected locations. Please set the CUDA_PATH environment "
        "variable to indicate the location of your CUDA.")
        NO_CUDA = True
    return setup_fpath, NO_CUDA, CUDA_PATH





def setup_cpu_fast_hadamard_extensions(setup_fpath):
    """Finds the paths for the fast hadamard transform extensions for CPU and
    sets them up."""
    cpu_fast_transform_path = os.path.join(setup_fpath, "exactPolynomial",
                "feature_generation", "cpu_feat_gen")
    os.chdir(cpu_fast_transform_path)
    sources = []
    for target_dir in ["poly_ops"]:
        os.chdir(target_dir)
        for fname in os.listdir():
            if not fname.endswith(".cpp"):
                continue
            sources.append(os.path.abspath(fname))
        os.chdir("..")

    os.chdir(setup_fpath)

    cpu_basic_op_wrapper = os.path.join(cpu_fast_transform_path,
                    "cpu_poly_feats.pyx")
    sources += [cpu_basic_op_wrapper]
    cpu_basic_op_ext = Extension("cpu_poly_feats",
                    sources = sources, language="c++",
                    include_dirs=[numpy.get_include(),
                            cpu_fast_transform_path])
    return [cpu_basic_op_ext], [cpu_basic_op_wrapper]




def setup_cuda_fast_hadamard_extensions(setup_fpath, CUDA_PATH, NO_CUDA = False):
    """Finds the paths for the fast hadamard transform extensions for Cuda and
    sets them up. The CUDA extension is a little more involved
    since it has to be compiled by NVCC and we have not yet
    made this cross-platform friendly."""

    #The GPU cudaHadamardTransform extension requires a little more work...
    #has to be compiled by NVCC. Right now we call a shell script to
    #arrange some details of the library construction. (TODO: needs to
    #be cross-platform!)
    if NO_CUDA:
        return [], []
    cuda_hadamard_path = os.path.join(setup_fpath, "exactPolynomial",
            "feature_generation", "gpu_feat_gen")
    os.chdir(cuda_hadamard_path)
    subprocess.run(["chmod", "+x", "nvcc_compile.sh"], check = True)
    subprocess.run(["./nvcc_compile.sh"], check = True)

    if "libarray_operations.a" not in os.listdir():
        os.chdir(setup_fpath)
        return [], []
    else:
        os.chdir(setup_fpath)

        cuda_basic_path = os.path.join(cuda_hadamard_path,
                            "cuda_poly_feats.pyx")
        cuda_basic_ext = Extension("cuda_poly_feats",
                sources=[cuda_basic_path],
                language="c++",
                libraries=["array_operations", "cudart_static"],
                library_dirs = [cuda_hadamard_path, os.path.join(CUDA_PATH,
                                    "lib64")],
                include_dirs = [numpy.get_include(), os.path.join(CUDA_PATH,
                                    "include")],
                extra_link_args = ["-lrt"],
                )
        return [cuda_basic_ext], [cuda_basic_path]




def main():
    """Builds the package, including all currently used extensions."""
    setup_fpath, NO_CUDA, CUDA_PATH = initial_checks()

    cpu_fht_ext, cpu_fht_files = setup_cpu_fast_hadamard_extensions(setup_fpath)
    fht_cuda_ext, gpu_fht_files = \
            setup_cuda_fast_hadamard_extensions(setup_fpath, CUDA_PATH, NO_CUDA)

    ext_modules = cpu_fht_ext + fht_cuda_ext


    if len(fht_cuda_ext) == 0:
        print("Construction of the cudaHadamardTransform extension failed! "
            "exactPolynomial will run in CPU-only mode.")

    for ext in ext_modules:
        ext.cython_directives = {'language_level': "3"}

    read_me = os.path.join(setup_fpath, "README.md")
    with open(read_me, "r") as fhandle:
        long_description = "".join(fhandle.readlines())

    setup(
        name="exactPolynomial",
        version=get_version(setup_fpath),
        packages=find_packages(),
        cmdclass = {"build_ext": build_ext},
        author="Jonathan Parkinson",
        author_email="jlparkinson1@gmail.com",
        description="Fitting an exact polynomial to large datasets using iterative methods",
        long_description = long_description,
        long_description_content_type="text/markdown",
        install_requires=["numpy>=1.10", "scipy>=1.7.0",
                    "cython>=0.10"],
        ext_modules = ext_modules,
        package_data={"": ["*.h", "*.c", "*.cu", "*.cpp",
                            "*.pyx", "*.sh"]}
        )


    if NO_CUDA:
        print("\n\nWARNING! the cuda installation was not found at "
            "the expected locations. exactPolynomial has been installed for CPU "
            "use ONLY. If you want to run on GPU, please set the "
            "CUDA_PATH environment variable to indicate the location "
            "of your CUDA installation and reinstall.\n\n")


    #Do some cleanup. (Really only matters when running setup.py develop.)
    os.chdir(os.path.join(setup_fpath, "exactPolynomial", "feature_generation",
                "gpu_feat_gen"))
    for gpu_fht_file in gpu_fht_files:
        compiled_cython_fname = os.path.basename(gpu_fht_file.replace(".pyx", ".cpp"))
        if compiled_cython_fname in os.listdir():
            os.remove(compiled_cython_fname)

    os.chdir(os.path.join(setup_fpath, "exactPolynomial", "feature_generation",
                "cpu_feat_gen"))

    for cpu_fht_file in cpu_fht_files:
        compiled_cython_fname = os.path.basename(cpu_fht_file.replace(".pyx", ".cpp"))
        if compiled_cython_fname in os.listdir():
            os.remove(compiled_cython_fname)

    os.chdir(os.path.join(setup_fpath, "exactPolynomial", "feature_generation",
                "gpu_feat_gen"))
    if "libarray_operations.a" in os.listdir():
        os.remove("libarray_operations.a")


if __name__ == "__main__":
    main()
