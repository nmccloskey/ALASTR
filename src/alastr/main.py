from datetime import datetime
import random
import numpy as np

from alastr.backend.etl.OutputManager import OutputManager
from alastr.utils.PipelineManager import PipelineManager
from alastr.backend.tools.logger import initialize_logger, terminate_logger, logger
from alastr.backend.tools.auxiliary import project_path, as_path
from alastr import __version__


def main():
    start_time = datetime.now()
    config_path = project_path(as_path("config.yaml"))

    OM = OutputManager()
    PM = PipelineManager(OM)

    out_dir = OM.output_dir
    initialize_logger(start_time, out_dir, program_name="ALASTR", version=__version__)

    random_seed = 99
    random.seed(random_seed)
    np.random.seed(random_seed)
    logger.info(f"Random seed set to {random_seed}")

    try:
        PM.run()
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
    finally:
        terminate_logger(
            input_dir=OM.input_dir,
            output_dir=out_dir,
            config_path=config_path,
            config=OM.config,
            start_time=start_time,
            program_name="ALASTR",
            version=__version__,
        )


if __name__ == "__main__":
    main()
