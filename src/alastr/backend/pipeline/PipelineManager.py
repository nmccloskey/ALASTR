from alastr.backend.etl.OutputManager import OutputManager, SQLDaemon
from alastr.backend.tools.logger import logger, _rel


class PipelineManager:
    _instance = None
    _initialized = False  # Track initialization

    def __new__(cls, OM: OutputManager):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance  # Always return the same instance

    def __init__(self, OM: OutputManager):
        if self._initialized:
            return  # Prevent re-initialization
    
    def convert_config():
        """
        convert either .xlsx to .yaml or vice versa
        (perhaps with helper functions)

        """

    def parse_config():
        """
        read .yaml
        save to self.config
        """
        
    def init_section():
        """
        Call OM, which calls SQLDaemon
            creates tables in db

        """
    
    def run_section():
        """
        input section, user-defined function (e.g., run_morphology)
        shuttle all samples through data generation
            and optionally, aggregation
        """

    # and then probably a bunch of wrapper functions which call OM, SQLDaemon, and EDADaemon