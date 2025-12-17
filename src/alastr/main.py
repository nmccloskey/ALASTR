from tqdm import tqdm
from alastr.backend.tools.logger import logger
from alastr.backend.etl.OutputManager import OutputManager
from alastr.utils.PipelineManager import PipelineManager


def main():
    """
    Main pipeline for processing and analyzing text samples.
    """
    try:
        OM = OutputManager()
        PM = PipelineManager(OM)

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
            
            if OM.cluster:
                for table_name in results:
                    OM.run_clustering(table_name, section)
            
            if OM.aggregate or OM.compare_groups:
                OM.run_aggregate_analyses(results, section)
            
            if OM.visualize:
                OM.generate_visuals(section)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
