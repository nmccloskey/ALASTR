import os
import re
import logging
logger = logging.getLogger("CustomLogger")

class Table:
    def __init__(self, OM, name, sheet_name, section, subdir, file_name, primary_keys, pivot=None):
        self.om = OM
        self.name = name
        self.sheet_name = sheet_name
        self.section = section
        self.subdir = subdir
        self.file_name = file_name
        self.file_base = re.sub(r'\.[^.]*$', '', self.file_name)
        self.file_path = os.path.join(self.om.output_dir, self.subdir)
        self.primary_keys = primary_keys
        self.source_fn = None
        self.family = None
        self.num_rows = 0
        self.num_cols = 0
        self.tags = []
        self.created_at = None
        self.fact = False
        self.aggregation_level = None
        self.granularity = None
        self.fact_table = None
        self.grouping_table = None
        self.pivot = pivot
        
    def get_subdir(self):
        """Retrieve subfolder(s) within \section."""
        return self.subdir

    def get_file_path(self):
        """Retrieve file path from root directory."""
        return self.file_path

    def get_pks(self):
        return self.primary_keys

    def get_data(self, columns="*", filters=None):
        if columns != "*":
            columns = self.primary_keys + columns
        return self.om.access_data(self.name, columns, filters)

    def update_data(self, data):
        self.om.update_database(self.name, data)
    
    def export_to_excel(self):
        self.om.export_sql_to_excel(self.name)
