import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "end-to-end-mlops"
AUTHOR_NAME = "nthaihoc"
SRC_REPO = "ccs"
AUTHOR_EMAIL = "thaihoc.ictu@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    description="A project end-to-end using MLOPs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_NAME}/{REPO_NAME}/",
    project_url={
        "Bug Tracker": f"https://github.com/{AUTHOR_NAME}/{REPO_NAME}/issues"
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
