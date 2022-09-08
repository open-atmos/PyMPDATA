from setuptools import setup

setup(
    use_scm_version={"local_scheme": lambda _: "", "version_scheme": "post-release"}
)
