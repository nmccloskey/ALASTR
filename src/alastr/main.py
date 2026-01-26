#!/usr/bin/env python3
from pathlib import Path
import random, numpy as np
from datetime import datetime
from alastr.backend.tools.logger import (
    get_root,
    set_root,
    logger,
    initialize_logger,
    terminate_logger,
)
from alastr.backend.tools.auxiliary import (
    project_path,
    load_config,
    find_files,
    as_path
)
from tqdm import tqdm
from alastr.backend.tools.logger import logger
from alastr.backend.etl.OutputManager import OutputManager
from alastr.utils.PipelineManager import PipelineManager
from alastr import __version__


def main():
    """
    Main pipeline for processing and analyzing text samples.
    """
    try:
        start_time = datetime.now()
        # timestamp = start_time.strftime("%y%m%d_%H%M")
        config_path = project_path(as_path("config.yaml"))

        OM = OutputManager()
        PM = PipelineManager(OM)
        out_dir = OM.output_dir

        initialize_logger(start_time, out_dir, program_name="ALASTR", version=__version__)
        logger.info("Logger initialized and early logs flushed.")

        random_seed = 99
        random.seed(random_seed)
        np.random.seed(random_seed)
        logger.info(f"Random seed set to {random_seed}")

        doc_ids = PM.run_preprocessing()

        for section in PM.analyses:
            logger.info(f"Running {section} analysis.")
            PM.sections[section].create_raw_data_tables()
                    
            for doc_id in tqdm(doc_ids, desc="Analyzing samples"):
                sample_data = PM.get_sample_data(doc_id)
                
                if not sample_data:
                    logger.warning(f"Skipping empty doc {doc_id}")
                    continue

                logger.info(f"Running {section} analysis for doc_id {doc_id}")
                results = PM.run_section(section, sample_data)
                
                for table_name, data in results.items():
                    OM.tables[table_name].update_data(data)
            
            for table_name in results:
                OM.tables[table_name].export_to_excel()            

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")


    finally:
        # Always finalize logging and metadata
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
