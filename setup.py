from setuptools import setup, find_packages
from pathlib import Path

# Read requirements.txt
def read_requirements():
    req_file = Path(__file__).parent / "requirements.txt"
    return req_file.read_text().splitlines() if req_file.exists() else []

setup(
    name="elasticas",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=read_requirements(),
    include_package_data=True,

    entry_points={
        "console_scripts": [
            "elasticas=elasticas.cli:main",
            "streamlit_elasticas=webapp.streamlit_app:main"
        ]
    },
)
