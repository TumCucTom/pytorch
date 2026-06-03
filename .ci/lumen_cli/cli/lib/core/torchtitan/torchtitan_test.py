import logging
import re
from pathlib import Path
from typing import Any

from cli.lib.common.cli_helper import BaseRunner
from cli.lib.common.pip_helper import pip_install_packages
from cli.lib.common.utils import working_directory
from cli.lib.core.torchtitan.lib import (
    clone_torchtitan,
    load_torchtitan_test_library,
    run_test_plan,
)


logger = logging.getLogger(__name__)


def _nightly_index_url() -> str:
    # torchao and torchcomms nightlies must match the CUDA toolchain of the
    # build. Read CUDA_STABLE from generate_binary_build_matrix.py, the single
    # source of truth for the stable CUDA version (e.g. "13.0" -> cu130), rather
    # than hardcoding the wheel channel here.
    matrix_script = (
        Path(__file__).resolve().parents[6]
        / ".github"
        / "scripts"
        / "generate_binary_build_matrix.py"
    )
    match = re.search(
        r'^CUDA_STABLE\s*=\s*"([^"]+)"', matrix_script.read_text(), re.MULTILINE
    )
    if not match:
        raise RuntimeError(f"Could not find CUDA_STABLE in {matrix_script}")
    cuda_stable = match.group(1).replace(".", "")
    return f"https://download.pytorch.org/whl/nightly/cu{cuda_stable}"


class TorchtitanTestRunner(BaseRunner):
    def __init__(self, args: Any):
        self.work_directory = "torchtitan"
        self.test_plan = args.test_plan

    def prepare(self):
        clone_torchtitan(dst=self.work_directory)
        # torchao and torchcomms nightlies are required by torchtitan
        pip_install_packages(
            packages=[
                "--pre",
                "torchao",
                "torchcomms",
                "--index-url",
                _nightly_index_url(),
            ],
        )
        with working_directory(self.work_directory):
            pip_install_packages(packages=["-e", "."])
            pip_install_packages(packages=["pytest", "pytest-cov"])

    def run(self):
        self.prepare()
        with working_directory(self.work_directory):
            run_test_plan(self.test_plan, load_torchtitan_test_library())
