import copy
import json
import logging
import os
import random
import sqlglot
import numpy as np

import swirl.embedding_utils as embedding_utils
from index_selection_evaluation.selection.candidate_generation import (
    candidates_per_query,
    syntactically_relevant_indexes,
)
from index_selection_evaluation.selection.cost_evaluation import CostEvaluation
from index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector
from index_selection_evaluation.selection.utils import get_utilized_indexes
from index_selection_evaluation.selection.workload import Query, Workload

from .workload_embedder import WorkloadEmbedder

QUERY_PATH = "query_files"


class WorkloadGenerator(object):
    def __init__(
        self, config, workload_columns, random_seed, database_name, experiment_id=None, filter_utilized_columns=None,
        external_workload=False, workload_path=None, test_external_workload=False, test_workload_path=None
    ):
        # assert config["benchmark"] in [
        #     "TPCH",
        #     "TPCDS",
        #     "JOB",
        # ], f"Benchmark '{config['benchmark']}' is currently not supported."

        # For create view statement differentiation
        self.experiment_id = experiment_id
        self.filter_utilized_columns = filter_utilized_columns

        self.rnd = random.Random()
        self.rnd.seed(random_seed)
        self.np_rnd = np.random.default_rng(seed=random_seed)

        self.workload_columns = workload_columns
        self.database_name = database_name

        # Check if using external workload
        self.external_workload = external_workload
        self.workload_path = workload_path

        # Check if using external workload for testing
        self.test_external_workload = test_external_workload
        self.test_workload_path = test_workload_path

        validation_instances = config["validation_testing"]["number_of_workloads"]
        test_instances = config["validation_testing"]["number_of_workloads"]

        if self.external_workload:
            logging.info(f"Using external workload from: {self.workload_path}")
            training_groups, training_query_data = self._load_external_workload_file(self.workload_path, role="train")

            test_groups, test_query_data = [], []
            if self.test_external_workload and self.test_workload_path:
                logging.info(f"Using separate external workload for testing from: {self.test_workload_path}")
                test_groups, test_query_data = self._load_external_workload_file(self.test_workload_path, role="test")

            self._initialize_external_workloads(
                config=config,
                validation_instances=validation_instances,
                test_instances=test_instances,
                training_groups=training_groups,
                training_query_data=training_query_data,
                test_groups=test_groups,
                test_query_data=test_query_data,
            )
        else:
            self.benchmark = config["benchmark"]
            self.number_of_query_classes = self._set_number_of_query_classes() # a class is a set of queries with the same structure
            self.excluded_query_classes = set(config["excluded_query_classes"])
            self.varying_frequencies = config["varying_frequencies"] #bool,True if generate workloads with different frequencies

            # self.query_texts is list of lists. Outer list for query classes, inner list for instances of this class.
            self.query_texts = self._retrieve_query_texts()
            self.query_classes = set(range(1, self.number_of_query_classes + 1))
            self.available_query_classes = self.query_classes - self.excluded_query_classes # all classes except user specified classes

            self.globally_indexable_columns = self._select_indexable_columns(self.filter_utilized_columns) # select columns that are used and can be indexed
        if not self.external_workload:
            if config["similar_workloads"] and config["unknown_queries"] == 0:
                # Todo: this branch can probably be removed
                assert self.varying_frequencies, "Similar workloads can only be created with varying frequencies."
                self.wl_validation = [None]
                self.wl_testing = [None]
                _, self.wl_validation[0], self.wl_testing[0] = self._generate_workloads(
                    0, validation_instances, test_instances, config["size"]
                )
                if config["query_class_change_frequency"] is None:
                    self.wl_training = self._generate_similar_workloads(config["training_instances"], config["size"])
                else:
                    self.wl_training = self._generate_similar_workloads_qccf(
                        config["training_instances"], config["size"], config["query_class_change_frequency"]
                    )
            elif config["unknown_queries"] > 0:
                assert (
                    config["validation_testing"]["unknown_query_probabilities"][-1] > 0
                ), "Query unknown_query_probabilities should be larger 0."

                # Get database connection parameters from environment variables
                db_host = os.getenv('DATABASE_HOST', 'localhost')
                db_port = os.getenv('DATABASE_PORT', '54321')
                embedder_connector = PostgresDatabaseConnector(self.database_name, autocommit=True, host=db_host, port=db_port)
                embedder = WorkloadEmbedder(
                    # Transform globally_indexable_columns to list of lists.
                    self.query_texts,
                    0,
                    embedder_connector,
                    [list(map(lambda x: [x], self.globally_indexable_columns))],
                    retrieve_plans=True,
                )
                self.unknown_query_classes = embedding_utils.which_queries_to_remove(
                    embedder.plans, config["unknown_queries"], random_seed
                )

                self.unknown_query_classes = frozenset(self.unknown_query_classes) - self.excluded_query_classes
                missing_classes = config["unknown_queries"] - len(self.unknown_query_classes)
                self.unknown_query_classes = self.unknown_query_classes | frozenset(
                    self.rnd.sample(self.available_query_classes - frozenset(self.unknown_query_classes), missing_classes)
                )
                assert len(self.unknown_query_classes) == config["unknown_queries"]

                self.known_query_classes = self.available_query_classes - frozenset(self.unknown_query_classes)
                embedder = None

                for query_class in self.excluded_query_classes:
                    assert query_class not in self.unknown_query_classes

                logging.critical(f"Global unknown query classes: {sorted(self.unknown_query_classes)}")
                logging.critical(f"Global known query classes: {sorted(self.known_query_classes)}")

                for unknown_query_probability in config["validation_testing"]["unknown_query_probabilities"]:
                    _, wl_validation, wl_testing = self._generate_workloads(
                        0,
                        validation_instances,
                        test_instances,
                        config["size"],
                        unknown_query_probability=unknown_query_probability,
                    )
                    self.wl_validation.append(wl_validation)
                    self.wl_testing.append(wl_testing)

                assert (
                    len(self.wl_validation)
                    == len(config["validation_testing"]["unknown_query_probabilities"])
                    == len(self.wl_testing)
                ), "Validation/Testing workloads length fail"

                # We are temporarily restricting the available query classes now to exclude certain classes for training
                original_available_query_classes = self.available_query_classes
                self.available_query_classes = self.known_query_classes

                if config["similar_workloads"]:
                    if config["query_class_change_frequency"] is not None:
                        logging.critical(
                            f"Similar workloads with query_class_change_frequency: {config['query_class_change_frequency']}"
                        )
                        self.wl_training = self._generate_similar_workloads_qccf(
                            config["training_instances"], config["size"], config["query_class_change_frequency"]
                        )
                    else:
                        self.wl_training = self._generate_similar_workloads(config["training_instances"], config["size"])
                else:
                    self.wl_training, _, _ = self._generate_workloads(config["training_instances"], 0, 0, config["size"])
                # We are removing the restriction now.
                self.available_query_classes = original_available_query_classes
            else:
                self.wl_validation = [None]
                self.wl_testing = [None]
                self.wl_training, self.wl_validation[0], self.wl_testing[0] = self._generate_workloads(
                    config["training_instances"], validation_instances, test_instances, config["size"]
                )

        if not self.external_workload:
            logging.critical(f"Sample training workloads: {self.rnd.sample(self.wl_training, 10)}")
        logging.info("Finished generating workloads.")

    def _initialize_external_workloads(
        self,
        config,
        validation_instances,
        test_instances,
        training_groups,
        training_query_data,
        test_groups,
        test_query_data,
    ):
        """Unify training and testing external workloads and prepare workload partitions."""

        self.training_external_workload_raw = training_groups
        self.test_external_workload_raw = test_groups
        self.training_external_query_data = training_query_data
        self.test_external_query_data = test_query_data

        self._build_global_query_catalog(training_query_data, test_query_data)

        # External workloads always derive observation spaces from union catalog
        self.benchmark = "EXTERNAL"
        self.excluded_query_classes = set()
        self.varying_frequencies = True

        self.globally_indexable_columns = self._select_indexable_columns(self.filter_utilized_columns)

        self.wl_validation = [[]]
        self.wl_testing = [[]]

        training_workloads = self._convert_grouped_workloads(training_groups, source_label="external_train")

        if not training_workloads:
            raise ValueError(
                "External training workload file did not contain any workload groups."
            )

        if test_groups:
            test_workloads = self._convert_grouped_workloads(test_groups, source_label="external_test")
            validation_subset = self._limit_workloads(test_workloads, validation_instances)
            testing_subset = self._limit_workloads(test_workloads, test_instances)

            if not validation_subset:
                validation_subset = list(test_workloads)
            if not testing_subset:
                testing_subset = list(test_workloads)

            self.wl_training = training_workloads
            self.wl_validation[0] = validation_subset
            self.wl_testing[0] = [self._clone_workload(workload) for workload in testing_subset]

            logging.info(
                "Prepared external workloads - Training groups: %d, Validation queries: %d, Testing queries: %d",
                len(self.wl_training),
                len(self.wl_validation[0]),
                len(self.wl_testing[0]),
            )
        else:
            logging.warning(
                "No separate test workload provided; using training workload file for validation/testing split."
            )
            total = len(training_workloads)
            v_n = min(validation_instances, total)
            t_n = min(test_instances, max(0, total - v_n))

            validation_subset = training_workloads[:v_n]
            testing_subset = training_workloads[v_n:v_n + t_n]
            training_subset = training_workloads[v_n + t_n:]

            self.wl_training = training_subset if training_subset else training_workloads
            self.wl_validation[0] = validation_subset if validation_subset else []
            self.wl_testing[0] = [self._clone_workload(workload) for workload in testing_subset] if testing_subset else []

            logging.info(
                "External workloads fallback split - Training: %d, Validation: %d, Testing: %d",
                len(self.wl_training),
                len(self.wl_validation[0]),
                len(self.wl_testing[0]),
            )

    def _convert_grouped_workloads(self, grouped_workloads, source_label):
        converted = []
        for wl_idx, wl_item in enumerate(grouped_workloads or []):
            queries = []
            for query_data in wl_item.get("queries", []):
                try:
                    query = self._build_query_from_external(query_data)
                except Exception as exc:
                    logging.error(
                        "Failed to convert query from %s workload at group %d: %s",
                        source_label,
                        wl_idx,
                        exc,
                    )
                    continue

                assert len(query.columns) > 0, f"Query columns should have length > 0: {query.text}"
                queries.append(query)

            if not queries:
                logging.warning(
                    "Skipping empty workload group %d from %s workload file.", wl_idx, source_label
                )
                continue

            description = wl_item.get("description", f"{source_label}_group_{wl_idx}")
            converted.append(Workload(queries, description=description, db=self.database_name))

        return converted

    @staticmethod
    def _limit_workloads(workloads, limit):
        if limit is None or limit <= 0:
            return list(workloads)
        return list(workloads[: limit]) if len(workloads) > limit else list(workloads)

    @staticmethod
    def _clone_workload(workload):
        cloned_queries = []
        for query in workload.queries:
            cloned_queries.append(
                Query(
                    query.nr,
                    query.text,
                    columns=list(query.columns),
                    frequency=query.frequency,
                )
            )
        clone = Workload(cloned_queries, description=workload.description, db=workload.db)
        clone.budget = workload.budget
        return clone

    def _build_query_from_external(self, query_data):
        sql_text = query_data["sql"]
        frequency = query_data.get("frequency", 1)
        tables_columns = query_data.get("tables_columns", {}) or {}

        signature = self._query_signature(sql_text, tables_columns)
        global_id = self._global_query_signature_to_id.get(signature)
        if global_id is None:
            # Register unseen query on-the-fly to maintain mapping consistency
            global_id = self._register_additional_query(sql_text, tables_columns)

        query_columns = self._columns_from_annotation(tables_columns)
        query = Query(global_id, sql_text, columns=query_columns, frequency=frequency)
        self._store_indexable_columns(query)
        return query

    def _columns_from_annotation(self, tables_columns):
        resolved_columns = []
        for table_name, column_names in tables_columns.items():
            for col_name in column_names:
                for wc in self.workload_columns:
                    if wc.table.name.lower() == str(table_name).lower() and wc.name.lower() == str(col_name).lower():
                        resolved_columns.append(wc)
                        break
        return resolved_columns

    def _build_global_query_catalog(self, training_query_data, test_query_data):
        self.external_query_data = []
        self._global_query_signature_to_id = {}

        all_query_records = []
        if training_query_data:
            all_query_records.extend(training_query_data)
        if test_query_data:
            all_query_records.extend(test_query_data)

        for record in all_query_records:
            sql_text = record["sql"]
            tables_columns = record.get("tables_columns", {}) or {}
            signature = self._query_signature(sql_text, tables_columns)

            if signature not in self._global_query_signature_to_id:
                global_id = len(self.external_query_data) + 1
                self._global_query_signature_to_id[signature] = global_id
                self.external_query_data.append(
                    {
                        "query_id": global_id,
                        "sql": sql_text,
                        "frequency": record.get("frequency", 1),
                        "tables_columns": tables_columns,
                    }
                )
            record["global_id"] = self._global_query_signature_to_id[signature]

        self.number_of_query_classes = len(self.external_query_data)
        self.query_classes = set(range(1, self.number_of_query_classes + 1))
        self.available_query_classes = self.query_classes.copy()

        if self.number_of_query_classes == 0:
            raise ValueError("No queries found across training/test external workloads.")

        self.query_texts = [[query_data["sql"]] for query_data in self.external_query_data]

    def _register_additional_query(self, sql_text, tables_columns):
        signature = self._query_signature(sql_text, tables_columns)
        if signature in self._global_query_signature_to_id:
            return self._global_query_signature_to_id[signature]

        global_id = len(self.external_query_data) + 1
        self._global_query_signature_to_id[signature] = global_id
        self.external_query_data.append(
            {
                "query_id": global_id,
                "sql": sql_text,
                "frequency": 1,
                "tables_columns": tables_columns,
            }
        )
        self.number_of_query_classes = len(self.external_query_data)
        self.query_classes = set(range(1, self.number_of_query_classes + 1))
        self.available_query_classes = self.query_classes.copy()
        self.query_texts = [[query_data["sql"]] for query_data in self.external_query_data]
        return global_id

    @staticmethod
    def _query_signature(sql_text, tables_columns):
        normalized_sql = " ".join(str(sql_text).split())
        normalized_columns = tuple(
            sorted(
                (
                    str(table_name).lower(),
                    tuple(sorted(str(column_name).lower() for column_name in column_names)),
                )
                for table_name, column_names in (tables_columns or {}).items()
            )
        )
        return normalized_sql, normalized_columns

    def _load_external_workload_file(self, workload_path, role):
        """Load and normalize an external workload JSON file."""
        try:
            with open(workload_path, 'r', encoding='utf-8') as f:
                workload_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"External workload file not found: {workload_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in workload file {workload_path}: {e}")

        normalized_groups = self._normalize_external_workload_structure(workload_data)

        query_data = []
        for wl_item in normalized_groups:
            for query in wl_item.get("queries", []):
                if "sql" not in query:
                    continue
                query_data.append(
                    {
                        "sql": query["sql"],
                        "frequency": query.get("frequency", 1),
                        "tables_columns": query.get("tables_columns", {}),
                        "source_role": role,
                    }
                )

        logging.info(
            "Loaded %d query definitions across %d workload groups from %s",
            len(query_data),
            len(normalized_groups),
            workload_path,
        )

        return normalized_groups, query_data

    @staticmethod
    def _normalize_external_workload_structure(workload_data):
        if isinstance(workload_data, list):
            return workload_data

        if isinstance(workload_data, dict):
            if isinstance(workload_data.get("workloads"), list):
                return workload_data["workloads"]
            if isinstance(workload_data.get("workload_groups"), list):
                flattened = []
                for group in workload_data["workload_groups"]:
                    flattened.extend(group.get("workloads", []))
                return flattened

        logging.warning("Unrecognized external workload JSON shape; expected list or structured dictionary.")
        return []

    def _set_number_of_query_classes(self):
        if self.external_workload:
            return len(self.external_query_data)
        else:
            if self.benchmark == "TPCH":
                return 22
            elif self.benchmark == "TPCDS":
                return 99
            elif self.benchmark == "JOB":
                return 113
            elif self.benchmark == "BASKETBALL":
                return 48
            elif self.benchmark == "BASEBALL":
                return 499
            elif self.benchmark == "CHEMBL":
                return 37
            elif self.benchmark == "DSB":
                return 53
            elif self.benchmark == "MIX3":
                return 120
            else:
                # raise ValueError("Unsupported Benchmark type provided, only TPCH, TPCDS, and JOB supported.")
                return 500

    def _retrieve_query_texts(self):
        query_files = [
            open(f"{QUERY_PATH}/{self.benchmark}/{self.benchmark}_{file_number}.txt", "r")
            for file_number in range(1, self.number_of_query_classes + 1)
        ]

        finished_queries = []
        for query_file in query_files:
            queries = query_file.readlines()[:1]
            queries = self._preprocess_queries(queries)

            finished_queries.append(queries)

            query_file.close()

        assert len(finished_queries) == self.number_of_query_classes

        return finished_queries

    def _preprocess_queries(self, queries):
        processed_queries = []
        for query in queries:
            query = query.replace("limit 100", "")
            query = query.replace("limit 20", "")
            query = query.replace("limit 10", "")
            query = query.strip()

            if "create view revenue0" in query:
                query = query.replace("revenue0", f"revenue0_{self.experiment_id}")

            processed_queries.append(query)

        return processed_queries

    def _store_indexable_columns(self, query):
        # Prefer SQLGlot parsing with alias-aware resolution; fallback to simple substring matching on failure
        try:
            parsed_sql = sqlglot.parse_one(query.text)

            # Build alias -> base table mapping and collect all base tables referenced in the query
            alias_to_table = {}
            referenced_tables = set()

            def _id_name(exp):
                try:
                    # Identifier-like objects generally expose .name; otherwise stringify
                    return getattr(exp, "name", str(exp))
                except Exception:
                    return str(exp)

            for table_exp in parsed_sql.find_all(sqlglot.exp.Table):
                try:
                    base_table = _id_name(table_exp.this)
                    referenced_tables.add(base_table)
                    alias_exp = table_exp.args.get("alias")
                    if alias_exp is not None and getattr(alias_exp, "this", None) is not None:
                        alias_name = _id_name(alias_exp.this)
                        alias_to_table[alias_name] = base_table
                except Exception:
                    # Be robust against dialect quirks; best-effort fallback to string form
                    base_table = str(table_exp)
                    referenced_tables.add(base_table)

            # Extract column usages with potential qualifiers (aliases)
            seen = set()
            for col_exp in parsed_sql.find_all(sqlglot.exp.Column):
                try:
                    col_name = _id_name(col_exp.this)
                except Exception:
                    col_name = str(col_exp)

                qualifier = None
                try:
                    qualifier_exp = col_exp.args.get("table")
                    if qualifier_exp is not None:
                        qualifier = _id_name(qualifier_exp)
                except Exception:
                    qualifier = None

                candidate_tables = []
                if qualifier:
                    # Map alias to real table if alias is present
                    base_table = alias_to_table.get(qualifier, qualifier)
                    candidate_tables = [base_table]
                else:
                    # Unqualified column: consider all tables present in the statement
                    candidate_tables = list(referenced_tables)

                for wc in self.workload_columns:
                    if wc.name != col_name:
                        continue
                    if wc.table.name in candidate_tables:
                        key = (wc.table.name, wc.name)
                        if key not in seen:
                            query.columns.append(wc)
                            seen.add(key)
        except Exception as e:
            logging.warning(f"SQLGlot parse failed for query {getattr(query, 'nr', '?')}: {e}. Falling back to substring matching.")
            if self.benchmark != "JOB":
                for column in self.workload_columns:
                    if column.name in query.text:
                        query.columns.append(column)
            else:
                query_text = query.text
                if "WHERE" not in query_text:
                    return

                split = query_text.split("WHERE")
                if len(split) != 2:
                    return
                query_text_before_where = split[0]
                query_text_after_where = split[1]

                for column in self.workload_columns:
                    if column.name in query_text_after_where and f"{column.table.name} " in query_text_before_where:
                        query.columns.append(column)
        


    def _workloads_from_tuples(self, tuples, unknown_query_probability=None):
        workloads = []
        unknown_query_probability = "" if unknown_query_probability is None else unknown_query_probability

        for tupl in tuples:
            query_classes, query_class_frequencies = tupl
            queries = []

            for query_class, frequency in zip(query_classes, query_class_frequencies):
                # For external workloads (both training and test), use the stored query data
                if (self.external_workload or self.test_external_workload) and hasattr(self, 'external_query_data'):
                    # Use the actual query data from external file
                    query_data = self.external_query_data[query_class - 1]
                    query_text = query_data['sql']
                    tables_columns = query_data.get('tables_columns', {})

                    # Create columns from tables_columns information in external workload
                    query_columns = []
                    for table_name, column_names in tables_columns.items():
                        for col_name in column_names:
                            # Find the corresponding column object from workload_columns
                            for wc in self.workload_columns:
                                if wc.table.name.lower() == table_name.lower() and wc.name.lower() == col_name.lower():
                                    query_columns.append(wc)
                                    break

                    # Use the frequency from the tuple (generated by internal logic) instead of external frequency
                    query = Query(query_class, query_text, columns=query_columns, frequency=frequency)
                else:
                    # For internal query classes, use the original logic
                    query_text = self.rnd.choice(self.query_texts[query_class - 1])
                    query = Query(query_class, query_text, frequency=frequency)

                self._store_indexable_columns(query)
                assert len(query.columns) > 0, f"Query columns should have length > 0: {query.text}"

                queries.append(query)

            assert isinstance(queries, list), f"Queries is not of type list but of {type(queries)}"
            previously_unseen_queries = (
                round(unknown_query_probability * len(queries)) if unknown_query_probability != "" else 0
            )
            workloads.append(
                Workload(queries, description=f"Contains {previously_unseen_queries} previously unseen queries.", db=self.database_name)
            )

        return workloads

    def _generate_workloads(
        self, train_instances, validation_instances, test_instances, size, unknown_query_probability=None
    ):
        required_unique_workloads = train_instances + validation_instances + test_instances

        unique_workload_tuples = set()

        while required_unique_workloads > len(unique_workload_tuples):
            workload_tuple = self._generate_random_workload(size, unknown_query_probability)
            unique_workload_tuples.add(workload_tuple)

        validation_tuples = self.rnd.sample(unique_workload_tuples, validation_instances)
        unique_workload_tuples = unique_workload_tuples - set(validation_tuples)

        test_workload_tuples = self.rnd.sample(unique_workload_tuples, test_instances)
        unique_workload_tuples = unique_workload_tuples - set(test_workload_tuples)

        assert len(unique_workload_tuples) == train_instances
        train_workload_tuples = unique_workload_tuples

        assert (
            len(train_workload_tuples) + len(test_workload_tuples) + len(validation_tuples) == required_unique_workloads
        )

        validation_workloads = self._workloads_from_tuples(validation_tuples, unknown_query_probability)
        test_workloads = self._workloads_from_tuples(test_workload_tuples, unknown_query_probability)
        train_workloads = self._workloads_from_tuples(train_workload_tuples, unknown_query_probability)

        return train_workloads, validation_workloads, test_workloads

    # The core idea is to create workloads that are similar and only change slightly from one to another.
    # For the following workload, we remove one random element, add another random one with frequency, and
    # randomly change the frequency of one element (including the new one).
    def _generate_similar_workloads(self, instances, size):
        assert size <= len(
            self.available_query_classes
        ), "Cannot generate workload with more queries than query classes"

        workload_tuples = []

        query_classes = self.rnd.sample(self.available_query_classes, size)
        available_query_classes = self.available_query_classes - frozenset(query_classes)
        frequencies = list(self.np_rnd.zipf(1.5, size))

        workload_tuples.append((copy.copy(query_classes), copy.copy(frequencies)))

        for workload_idx in range(instances - 1):
            # Remove a random element
            idx_to_remove = self.rnd.randrange(len(query_classes))
            query_classes.pop(idx_to_remove)
            frequencies.pop(idx_to_remove)

            # Draw a new random element, the removed one is excluded
            query_classes.append(self.rnd.sample(available_query_classes, 1)[0])
            frequencies.append(self.np_rnd.zipf(1.5, 1)[0])

            frequencies[self.rnd.randrange(len(query_classes))] = self.np_rnd.zipf(1.5, 1)[0]

            available_query_classes = self.available_query_classes - frozenset(query_classes)
            workload_tuples.append((copy.copy(query_classes), copy.copy(frequencies)))

        workloads = self._workloads_from_tuples(workload_tuples)

        return workloads

    # This version uses the same query id selction for query_class_change_frequency workloads
    def _generate_similar_workloads_qccf(self, instances, size, query_class_change_frequency):
        assert size <= len(
            self.available_query_classes
        ), "Cannot generate workload with more queries than query classes"

        workload_tuples = []

        while len(workload_tuples) < instances:
            if len(workload_tuples) % query_class_change_frequency == 0:
                query_classes = self.rnd.sample(self.available_query_classes, size)

            frequencies = list(self.np_rnd.integers(1, 30, size))
            workload_tuples.append((copy.copy(query_classes), copy.copy(frequencies)))

        workloads = self._workloads_from_tuples(workload_tuples)

        return workloads

    def _generate_random_workload(self, size, unknown_query_probability=None):
        assert size <= self.number_of_query_classes, "Cannot generate workload with more queries than query classes"

        workload_query_classes = None
        if unknown_query_probability is not None:
            number_of_unknown_queries = round(size * unknown_query_probability)
            number_of_known_queries = size - number_of_unknown_queries
            assert number_of_known_queries + number_of_unknown_queries == size

            known_query_classes = self.rnd.sample(self.known_query_classes, number_of_known_queries)
            unknown_query_classes = self.rnd.sample(self.unknown_query_classes, number_of_unknown_queries)
            query_classes = known_query_classes
            query_classes.extend(unknown_query_classes)
            workload_query_classes = tuple(query_classes)
            assert len(workload_query_classes) == size
        else:
            workload_query_classes = tuple(self.rnd.sample(self.available_query_classes, size))

        # Create frequencies
        if self.varying_frequencies:
            query_class_frequencies = tuple(list(self.np_rnd.integers(1, 30, size)))
        else:
            query_class_frequencies = tuple([1 for frequency in range(size)])

        workload_tuple = (workload_query_classes, query_class_frequencies)

        return workload_tuple

    def _only_utilized_indexes(self, indexable_columns):#select columns that are used and can be indexed
        frequencies = [1 for frequency in range(len(self.available_query_classes))]
        workload_tuple = (self.available_query_classes, frequencies)
        workload = self._workloads_from_tuples([workload_tuple])[0]

        candidates = candidates_per_query(
            workload,
            max_index_width=1,
            candidate_generator=syntactically_relevant_indexes,
        )

        # Get database connection parameters from environment variables
        db_host = os.getenv('DATABASE_HOST', 'localhost')
        db_port = os.getenv('DATABASE_PORT', '54321')
        connector = PostgresDatabaseConnector(self.database_name, autocommit=True, host=db_host, port=db_port)
        connector.drop_indexes()
        cost_evaluation = CostEvaluation(connector)

        utilized_indexes, query_details = get_utilized_indexes(workload, candidates, cost_evaluation, True)

        columns_of_utilized_indexes = set()
        for utilized_index in utilized_indexes:
            column = utilized_index.columns[0]
            columns_of_utilized_indexes.add(column)

        output_columns = columns_of_utilized_indexes & set(indexable_columns)
        excluded_columns = set(indexable_columns) - output_columns
        logging.critical(f"Excluding columns based on utilization:\n   {excluded_columns}")

        return output_columns

    def _load_external_workload(self):
        """Load workload from external JSON file and convert to internal query class format"""
        try:
            with open(self.workload_path, 'r', encoding='utf-8') as f:
                workload_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"External workload file not found: {self.workload_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in workload file: {e}")

        # Preserve raw grouped workloads (support list or {'workloads': [...]})
        if isinstance(workload_data, dict) and 'workloads' in workload_data and isinstance(workload_data['workloads'], list):
            self.external_workload_raw = workload_data['workloads']
        elif isinstance(workload_data, list):
            self.external_workload_raw = workload_data
        else:
            logging.warning("Unrecognized external workload JSON shape; expected list or {'workloads': [...]}.")
            self.external_workload_raw = []

        # Extract all unique SQL queries and their frequencies from external workloads
        self.external_query_data = []
        query_id = 1

        # Iterate over normalized raw groups to support both list and {'workloads': [...]} shapes
        for wl_item in self.external_workload_raw:
            for query_data in wl_item.get("queries", []):
                sql_text = query_data["sql"]
                frequency = query_data.get("frequency", 1)
                tables_columns = query_data.get("tables_columns", {})

                # Store query data for later processing
                self.external_query_data.append({
                    'query_id': query_id,
                    'sql': sql_text,
                    'frequency': frequency,
                    'tables_columns': tables_columns
                })
                query_id += 1

        logging.info(f"Loaded {len(self.external_query_data)} queries from external file")

        # Initialize required attributes for external workloads (similar to internal query classes)
        self.benchmark = "EXTERNAL"
        self.number_of_query_classes = len(self.external_query_data)
        self.query_classes = set(range(1, self.number_of_query_classes + 1))
        self.available_query_classes = self.query_classes.copy()
        self.excluded_query_classes = set()
        self.varying_frequencies = True
        
        # Create query_texts in the same format as internal query classes
        # Each query class contains only one query (the external query)
        self.query_texts = [[query_data['sql']] for query_data in self.external_query_data]

        # Set up globally indexable columns (will be processed later like internal queries)
        self.globally_indexable_columns = self._select_indexable_columns(self.filter_utilized_columns)

    def _load_test_external_workload(self):
        """Load test workload from external JSON file and convert to internal query class format"""
        try:
            with open(self.test_workload_path, 'r', encoding='utf-8') as f:
                workload_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Test external workload file not found: {self.test_workload_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in test workload file: {e}")

        # Preserve raw grouped test workloads (support list or {'workloads': [...]})
        if isinstance(workload_data, dict) and 'workloads' in workload_data and isinstance(workload_data['workloads'], list):
            self.test_external_workload_raw = workload_data['workloads']
        elif isinstance(workload_data, list):
            self.test_external_workload_raw = workload_data
        else:
            logging.warning("Unrecognized test external workload JSON shape; expected list or {'workloads': [...]}.")
            self.test_external_workload_raw = []

        # Extract all unique SQL queries and their frequencies from test external workloads
        self.test_external_query_data = []
        query_id = 1

        # Iterate over normalized raw groups to support both list and {'workloads': [...]} shapes
        for wl_item in self.test_external_workload_raw:
            for query_data in wl_item.get("queries", []):
                sql_text = query_data["sql"]
                frequency = query_data.get("frequency", 1)
                tables_columns = query_data.get("tables_columns", {})

                # Store query data for later processing
                self.test_external_query_data.append({
                    'query_id': query_id,
                    'sql': sql_text,
                    'frequency': frequency,
                    'tables_columns': tables_columns
                })
                query_id += 1

        logging.info(f"Loaded {len(self.test_external_query_data)} test queries from external file")



    def _select_indexable_columns(self, only_utilized_indexes=False):
        schema_columns = list(self.workload_columns)
        logging.info(
            "Selecting indexable columns from schema definitions%s."
            % (" with utilization filtering" if only_utilized_indexes else "")
        )

        if only_utilized_indexes:
            indexable_columns = self._only_utilized_indexes(schema_columns)
        else:
            indexable_columns = schema_columns

        indexable_column_set = set(indexable_columns)
        selected_columns = []

        for column in schema_columns:
            column.global_column_id = None

        global_column_id = 0
        for column in schema_columns:
            if column in indexable_column_set:
                column.global_column_id = global_column_id
                global_column_id += 1
                selected_columns.append(column)

        logging.info(f"Selected {len(selected_columns)} schema columns as indexable candidates.")

        return selected_columns