import logging
import os
import platform
import re
import subprocess
import sqlglot

from .utils import b_to_mb
from .workload import Column, Table


class TableGenerator:
    def __init__(
        self,
        benchmark_name,
        scale_factor,
        database_connector,
        explicit_database_name=None,
    ):
        self.scale_factor = scale_factor
        self.benchmark_name = benchmark_name
        self.db_connector = database_connector
        self.explicit_database_name = explicit_database_name

        self.database_names = self.db_connector.database_names()
        self.tables = []
        self.columns = []
        self._prepare()
        # 只在非JOB、非AIRLINE和非BASEBALL时才自动建库建表
        if self.benchmark_name in ["tpch", "tpcds"]:
            self.database_names = self.db_connector.database_names()
            if self.database_name() not in self.database_names:
                self._generate()
                self.create_database()
            else:
                logging.debug("Database with given scale factor already existing")
        # 只需读取表结构
        self._read_column_names()



    def _read_column_names(self):
        # Read table and column names from 'create table' statements
        filename = self.directory + "/" + self.create_table_statements_file
        with open(filename, "r") as file:
            data = file.read().lower()
        create_tables = data.split("create table ")[1:]
        for create_table in create_tables:
            splitted = create_table.split("(", 1)
            if len(splitted) < 2:
                continue
            table = Table(splitted[0].strip())
            self.tables.append(table)
            
            # 找到表的结束位置（第一个分号）
            table_content = splitted[1]
            end_pos = table_content.find(");")
            if end_pos != -1:
                table_content = table_content[:end_pos]
            
            # 分割列定义
            columns = table_content.split(",\n")
            for column in columns:
                column = column.strip()
                if not column or column.startswith("primary key"):
                    continue
                # 提取列名（第一个空格前的部分）
                name = column.split(" ", 1)[0].strip()
                if name:
                    column_object = Column(name)
                    table.add_column(column_object)
                    self.columns.append(column_object)
            

    def _generate(self):
        logging.info("Generating {} data".format(self.benchmark_name))
        logging.info("scale factor: {}".format(self.scale_factor))
        self._run_make()
        
        # 对于不需要生成数据的benchmark（如ACCIDENTS），跳过数据生成步骤
        if self.cmd is not None:
            self._run_command(self.cmd)
            if self.benchmark_name == "tpcds":
                self._run_command(["bash", "../../scripts/replace_in_dat.sh"])
            logging.info("[Generate command] " + " ".join(self.cmd))
        else:
            logging.info("No data generation needed for this benchmark")
            
        self._table_files()
        logging.info("Files generated: {}".format(self.table_files))

    def create_database(self):
        self.db_connector.create_database(self.database_name())
        filename = self.directory + "/" + self.create_table_statements_file
        with open(filename, "r") as file:
            create_statements = file.read()
        # Do not create primary keys
        create_statements = re.sub(r",\s*primary key (.*)", "", create_statements)
        self.db_connector.db_name = self.database_name()
        self.db_connector.create_connection()
        self.create_tables(create_statements)
        self._load_table_data(self.db_connector)
        self.db_connector.enable_simulation()

    def create_tables(self, create_statements):
        logging.info("Creating tables")
        for create_statement in create_statements.split(";")[:-1]:
            self.db_connector.exec_only(create_statement)
        self.db_connector.commit()

    def _load_table_data(self, database_connector):
        logging.info("Loading data into the tables")
        if not self.table_files:
            logging.info("No data files to load for this benchmark")
            return
            
        for filename in self.table_files:
            logging.debug("    Loading file {}".format(filename))
            table = filename.replace(".tbl", "").replace(".dat", "")
            path = self.directory + "/" + filename
            size = os.path.getsize(path)
            size_string = f"{b_to_mb(size):,.4f} MB"
            logging.debug(f"    Import data of size {size_string}")
            database_connector.import_data(table, path)
            os.remove(os.path.join(self.directory, filename))
        database_connector.commit()

    def _run_make(self):
        if self.make_command is None:
            logging.info("No make command needed for this benchmark")
            return
            
        if "dbgen" not in self._files() and "dsdgen" not in self._files():
            logging.info("Running make in {}".format(self.directory))
            self._run_command(self.make_command)
        else:
            logging.info("No need to run make")

    def _table_files(self):
        self.table_files = [x for x in self._files() if ".tbl" in x or ".dat" in x or ".csv" in x]

    def _run_command(self, command):
        if command is None:
            logging.info("No command to run")
            return
            
        cmd_out = "[SUBPROCESS OUTPUT] "
        p = subprocess.Popen(
            command,
            cwd=self.directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with p.stdout:
            for line in p.stdout:
                logging.info(cmd_out + line.decode("utf-8").replace("\n", ""))
        p.wait()

    def _files(self):
        return os.listdir(self.directory)

    def database_name(self):
        # JOB直接返回你实际的数据库名
        if self.benchmark_name == "job":
            return "imdb"  # 这里填写你实际的数据库名
        elif self.benchmark_name == "baseball":
            return "baseball"
        elif self.benchmark_name == "carcinogenesis":
            return "carcinogenesis"
        elif self.benchmark_name == "ccs":
            return "ccs"
        elif self.benchmark_name == "chembl":
            return "chembl"
        if self.explicit_database_name:
            return self.explicit_database_name
        # name = "indexselection_" + self.benchmark_name + "___"
        # name += str(self.scale_factor).replace(".", "_")
        return self.benchmark_name

    def _prepare(self):
        if self.benchmark_name == "tpch":
            self.make_command = ["make", "DATABASE=POSTGRESQL"]
            if platform.system() == "Darwin":
                self.make_command.append("MACHINE=MACOS")

            self.directory = "./index_selection_evaluation/tpch-kit/dbgen"
            self.create_table_statements_file = "dss.ddl"
            self.cmd = ["./dbgen", "-s", str(self.scale_factor), "-f"]
        elif self.benchmark_name == "tpcds":
            self.make_command = ["make"]
            sysname = platform.system()
            if sysname == "Darwin":
                self.make_command.append("OS=MACOS")
            elif sysname == "Linux":
                # 确保为 Linux 编译，并为 GCC10+ 添加 -fcommon，同时不要覆盖默认 CFLAGS
                # 通过覆盖 LINUX_CFLAGS 来追加 -fcommon，避免丢失 -D$(OS) 与基础标志
                self.make_command.append("OS=LINUX")
                self.make_command.append("LINUX_CFLAGS=-g -Wall -fcommon -fPIE")
                # 避免链接阶段 PIE 报错，禁用 PIE 链接
                self.make_command.append("LDFLAGS=-no-pie")

            self.directory = "./index_selection_evaluation/tpcds-kit/tools"
            self.create_table_statements_file = "tpcds.sql"
            self.cmd = ["./dsdgen", "-SCALE", str(self.scale_factor), "-FORCE"]

            # 0.001 is allowed for testing
            if (
                int(self.scale_factor) - self.scale_factor != 0
                and self.scale_factor != 0.001
            ):
                raise Exception("Wrong TPCDS scale factor")
        else:
            base_dir = "./query_files/"
            self.directory = base_dir+self.benchmark_name.upper()
            self.create_table_statements_file = "{}.sql".format(self.benchmark_name.upper())
            self.make_command = None
            self.cmd = None
       