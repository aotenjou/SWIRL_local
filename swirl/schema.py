import importlib
import logging
import os

from index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector
from index_selection_evaluation.selection.table_generator import TableGenerator


class Schema(object):
    def __init__(self, benchmark_name, scale_factor, filters={}):
        # Get database connection parameters from environment variables
        db_host = os.getenv('DATABASE_HOST', 'localhost')
        db_port = os.getenv('DATABASE_PORT', '54321')
        
        generating_connector = PostgresDatabaseConnector(None, autocommit=True, host=db_host, port=db_port)
        table_generator = TableGenerator(
            benchmark_name=benchmark_name.lower(), scale_factor=scale_factor, database_connector=generating_connector
        )

        self.database_name = table_generator.database_name()
        self.tables = table_generator.tables

        self.columns = []
        for table in self.tables:
            for column in table.columns:
                self.columns.append(column)

        for filter_name in filters.keys():
            filter_class = getattr(importlib.import_module("swirl.schema"), filter_name)
            filter_instance = filter_class(filters[filter_name], self.database_name)
            self.columns = filter_instance.apply_filter(self.columns)


class TableNumRowsFilter(object):
    def __init__(self, threshold, database_name):
        self.threshold = threshold
        # Get database connection parameters from environment variables
        db_host = os.getenv('DATABASE_HOST', 'localhost')
        db_port = os.getenv('DATABASE_PORT', '54321')
        
        self.connector = PostgresDatabaseConnector(database_name, autocommit=True, host=db_host, port=db_port)
        self.connector.create_statistics()

    def apply_filter(self, columns):
        output_columns = []

        for column in columns:
            table_name = column.table.name
            try:
                result = self.connector.exec_fetch(
                    f"SELECT reltuples::bigint AS estimate FROM pg_class where relname='{table_name}'"
                )
                
                # 检查结果是否为空
                if result is None or len(result) == 0:
                    logging.warning(f"Table {table_name} not found in database, skipping column filter")
                    output_columns.append(column)
                    continue
                
                table_num_rows = result[0]
                
                if table_num_rows > self.threshold:
                    output_columns.append(column)
            except Exception as e:
                logging.warning(f"Error checking table {table_name}: {e}, including column in filter")
                output_columns.append(column)

        logging.warning(f"Reduced columns from {len(columns)} to {len(output_columns)}.")

        return output_columns
