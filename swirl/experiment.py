import datetime
import gzip
import importlib
import json
import logging
import os
import pickle
import random
import subprocess
import time

import gym
import numpy as np

from gym_db.common import EnvironmentType
from index_selection_evaluation.selection.algorithms.db2advis_algorithm import DB2AdvisAlgorithm
from index_selection_evaluation.selection.algorithms.extend_algorithm import ExtendAlgorithm
from index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector

from . import utils
from .configuration_parser import ConfigurationParser
from .schema import Schema
from .workload_generator import WorkloadGenerator


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class Experiment(object):
    def __init__(self, configuration_file):
        self._init_times()

        cp = ConfigurationParser(configuration_file)
        self.config = cp.config
        self._set_sb_version_specific_methods()

        self.id = self.config["id"]
        self.model = None

        self.rnd = random.Random()
        self.rnd.seed(self.config["random_seed"])

        self.comparison_performances = {
            "test": {"Extend": [], "DB2Adv": []},
            "validation": {"Extend": [], "DB2Adv": []},
        }
        
        # 初始化时间记录变量
        self.swirl_times = {"test": [], "validation": []}
        self.extend_times = {"test": [], "validation": []}
        self.db2advis_times = {"test": [], "validation": []}
        self.comparison_indexes = {"Extend": set(), "DB2Adv": set()}

        self.number_of_features = None
        self.number_of_actions = None
        self.evaluated_workloads_strs = []
        # 存储workload的推荐索引labels
        self.workload_labels = {"validation": {}, "test": {}}

        self.EXPERIMENT_RESULT_PATH = self.config["result_path"]
        self._create_experiment_folder()

    def prepare(self):
        self.schema = Schema(
            self.config["workload"]["benchmark"],
            self.config["workload"]["scale_factor"],
            self.config["column_filters"],
        )

        self.workload_generator = WorkloadGenerator(
            self.config["workload"],
            workload_columns=self.schema.columns,
            random_seed=self.config["random_seed"],
            database_name=self.schema.database_name,
            experiment_id=self.id,
            filter_utilized_columns=self.config["filter_utilized_columns"],
            external_workload=self.config.get("ExternalWorkload", False),
            workload_path=self.config.get("WorkloadPath", None),
        )
        self._assign_budgets_to_workloads()

        self.globally_indexable_columns = self.workload_generator.globally_indexable_columns

        # [[single column indexes], [2-column combinations], [3-column combinations]...]
        self.globally_indexable_columns = utils.create_column_permutation_indexes(
            self.globally_indexable_columns, self.config["max_index_width"]
        )

        self.single_column_flat_set = set(map(lambda x: x[0], self.globally_indexable_columns[0]))

        self.globally_indexable_columns_flat = [item for sublist in self.globally_indexable_columns for item in sublist]
        logging.info(f"Feeding {len(self.globally_indexable_columns_flat)} candidates into the environments.")

        self.action_storage_consumptions = utils.predict_index_sizes(
            self.globally_indexable_columns_flat, self.schema.database_name
        )

        if "workload_embedder" in self.config:
            workload_embedder_class = getattr(
                importlib.import_module("swirl.workload_embedder"), self.config["workload_embedder"]["type"]
            )
            # Get database connection parameters from environment variables
            db_host = os.getenv('DATABASE_HOST', 'localhost')
            db_port = os.getenv('DATABASE_PORT', '54321')
            workload_embedder_connector = PostgresDatabaseConnector(self.schema.database_name, autocommit=True, host=db_host, port=db_port)
            self.workload_embedder = workload_embedder_class(
                self.workload_generator.query_texts,
                self.config["workload_embedder"]["representation_size"],
                workload_embedder_connector,
                self.globally_indexable_columns,
            )

        self.multi_validation_wl = []
        if len(self.workload_generator.wl_validation) > 1:
            for workloads in self.workload_generator.wl_validation:
                self.multi_validation_wl.extend(self.rnd.sample(workloads, min(7, len(workloads))))

    def _assign_budgets_to_workloads(self):
        for workload_list in self.workload_generator.wl_testing:
            for workload in workload_list:
                workload.budget = self.rnd.choice(self.config["budgets"]["validation_and_testing"])

        for workload_list in self.workload_generator.wl_validation:
            for workload in workload_list:
                workload.budget = self.rnd.choice(self.config["budgets"]["validation_and_testing"])

    def _save_workloads_as_json(self):
        """将workloads保存为JSON格式而不是pickle格式"""
        import json
        from datetime import datetime

        print("🔍 DEBUG: 开始保存workloads为JSON格式")
        print(f"🔍 DEBUG: 当前workload_labels状态: test={len(self.workload_labels.get('test', {}))}, validation={len(self.workload_labels.get('validation', {}))}")
        logging.info("开始保存workloads为JSON格式")
        logging.info(f"当前workload_labels状态: test={len(self.workload_labels.get('test', {}))}, validation={len(self.workload_labels.get('validation', {}))}")

        # 保存testing workloads
        if self.workload_generator.wl_testing:
            logging.info(f"处理testing workloads: {len(self.workload_generator.wl_testing)} 组")

            # 优先使用更新后的workload对象（包含labels），如果没有则使用原始对象
            if self.workload_labels.get("test"):
                logging.info(f"使用更新后的testing workloads (包含labels)")
                testing_workloads = self._get_updated_workloads("test")
            else:
                logging.info(f"使用原始testing workloads (无labels)")
                testing_workloads = self.workload_generator.wl_testing

            testing_data = self._serialize_workloads_to_json(testing_workloads)
            testing_file = f"{self.experiment_folder_path}/testing_workloads.json"
            with open(testing_file, "w", encoding='utf-8') as f:
                json.dump(testing_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            logging.info(f"✅ Testing workloads已保存到: {testing_file}")

        # 保存validation workloads
        if self.workload_generator.wl_validation:
            logging.info(f"处理validation workloads: {len(self.workload_generator.wl_validation)} 组")

            # 优先使用更新后的workload对象（包含labels），如果没有则使用原始对象
            if self.workload_labels.get("validation"):
                logging.info(f"使用更新后的validation workloads (包含labels)")
                validation_workloads = self._get_updated_workloads("validation")
            else:
                logging.info(f"使用原始validation workloads (无labels)")
                validation_workloads = self.workload_generator.wl_validation

            validation_data = self._serialize_workloads_to_json(validation_workloads)
            validation_file = f"{self.experiment_folder_path}/validation_workloads.json"
            with open(validation_file, "w", encoding='utf-8') as f:
                json.dump(validation_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            logging.info(f"✅ Validation workloads已保存到: {validation_file}")

        logging.info("JSON保存过程完成")

    def _get_updated_workloads(self, run_type):
        """从workload_labels中获取更新后的workload对象，并按原始结构组织"""
        logging.info(f"开始获取 {run_type} 的更新后workloads")

        if run_type not in self.workload_labels:
            logging.warning(f"{run_type} 不在workload_labels中，返回空列表")
            return []

        # 获取原始workload结构
        original_workloads = (self.workload_generator.wl_testing
                            if run_type == "test"
                            else self.workload_generator.wl_validation)

        logging.info(f"原始 {run_type} workloads: {len(original_workloads)} 组")

        # 创建更新后的workload结构
        updated_workloads = []
        total_original_workloads = 0
        total_updated_workloads = 0

        for i, workload_list in enumerate(original_workloads):
            updated_workload_list = []
            logging.info(f"处理原始组 {i}: {len(workload_list)} 个workloads")

            for workload in workload_list:
                total_original_workloads += 1
                # 查找对应的更新后workload
                updated_workload = self._find_updated_workload(workload, run_type)
                if updated_workload:
                    updated_workload_list.append(updated_workload)
                    total_updated_workloads += 1
                    logging.info(f"  ✅ 找到更新后的workload: {workload.description} (labels: {len(updated_workload.labels)})")
                else:
                    # 如果找不到更新后的workload，使用原始workload
                    logging.warning(f"  ❌ 找不到更新后的workload: {workload.description}，使用原始workload")
                    updated_workload_list.append(workload)
            updated_workloads.append(updated_workload_list)

        logging.info(f"✅ {run_type} workloads更新完成:")
        logging.info(f"   原始workloads总数: {total_original_workloads}")
        logging.info(f"   更新后workloads总数: {total_updated_workloads}")
        logging.info(f"   更新覆盖率: {total_updated_workloads/total_original_workloads*100:.1f}%")

        return updated_workloads

    def _find_updated_workload(self, original_workload, run_type):
        """在workload_labels中查找对应的更新后workload"""
        for workload_key, data in self.workload_labels[run_type].items():
            updated_workload = data["workload"]
            # 通过description匹配workload（可能需要更精确的匹配逻辑）
            if (updated_workload.description == original_workload.description and
                updated_workload.db == original_workload.db and
                updated_workload.id == original_workload.id):
                return updated_workload
        return None

    def _serialize_workloads_to_json(self, workload_lists):
        """将workload列表序列化为JSON格式"""
        from datetime import datetime

        serialized_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "experiment_id": self.id,
                "total_workload_groups": len(workload_lists)
            },
            "workload_groups": []
        }

        for i, workload_list in enumerate(workload_lists):
            group_data = {
                "group_id": i,
                "workloads": []
            }

            for workload in workload_list:
                workload_data = self._serialize_single_workload(workload)
                group_data["workloads"].append(workload_data)

            serialized_data["workload_groups"].append(group_data)

        return serialized_data

    def _serialize_single_workload(self, workload):
        """序列化单个Workload对象"""
        # 序列化queries
        queries_data = []
        for query in workload.queries:
            query_data = {
                "id": query.nr,
                "text": query.text,
                "frequency": query.frequency,
                "columns": [
                    {
                        "name": column.name,
                        "table": column.table.name if column.table else None,
                        "global_column_id": column.global_column_id
                    } for column in query.columns
                ]
            }
            queries_data.append(query_data)

        # 序列化labels (Index对象)
        labels_data = []
        for index in workload.labels:
            index_data = {
                "table": index.table().name,
                "columns": [col.name for col in index.columns],
                "estimated_size": index.estimated_size,
                "hypopg_name": index.hypopg_name
            }
            labels_data.append(index_data)

        # 构建完整的workload数据
        workload_data = {
            "db": workload.db,
            "id": workload.id,
            "description": workload.description,
            "budget": workload.budget,
            "queries": queries_data,
            "labels": labels_data,
            "labels_count": len(workload.labels),
            "queries_count": len(workload.queries)
        }

        return workload_data

    def finish(self):
        self.end_time = datetime.datetime.now()

        self.model.training = False
        self.model.env.norm_reward = False
        self.model.env.training = False

        self.test_fm = self.test_model(self.model)[0]
        self.vali_fm = self.validate_model(self.model)[0]

        # Load moving average model if it exists
        moving_average_model_path = f"{self.experiment_folder_path}/moving_average_model.zip"
        if os.path.exists(moving_average_model_path):
            self.moving_average_model = self.model_type.load(moving_average_model_path)
            self.moving_average_model.training = False
            self.test_ma = self.test_model(self.moving_average_model)[0]
            self.vali_ma = self.validate_model(self.moving_average_model)[0]
        else:
            logging.warning(f"Moving average model not found at {moving_average_model_path}, skipping evaluation")
            self.moving_average_model = None
            self.test_ma = None
            self.vali_ma = None
            
        if len(self.multi_validation_wl) > 0:
            moving_average_model_mv_path = f"{self.experiment_folder_path}/moving_average_model_mv.zip"
            if os.path.exists(moving_average_model_mv_path):
                self.moving_average_model_mv = self.model_type.load(moving_average_model_mv_path)
                self.moving_average_model_mv.training = False
                self.test_ma_mv = self.test_model(self.moving_average_model_mv)[0]
                self.vali_ma_mv = self.validate_model(self.moving_average_model_mv)[0]
            else:
                logging.warning(f"Moving average model MV not found at {moving_average_model_mv_path}, skipping evaluation")
                self.moving_average_model_mv = None
                self.test_ma_mv = None
                self.vali_ma_mv = None

        # Load moving average model 3 if it exists
        moving_average_model_3_path = f"{self.experiment_folder_path}/moving_average_model_3.zip"
        if os.path.exists(moving_average_model_3_path):
            self.moving_average_model_3 = self.model_type.load(moving_average_model_3_path)
            self.moving_average_model_3.training = False
            self.test_ma_3 = self.test_model(self.moving_average_model_3)[0]
            self.vali_ma_3 = self.validate_model(self.moving_average_model_3)[0]
        else:
            logging.warning(f"Moving average model 3 not found at {moving_average_model_3_path}, skipping evaluation")
            self.moving_average_model_3 = None
            self.test_ma_3 = None
            self.vali_ma_3 = None
            
        if len(self.multi_validation_wl) > 0:
            moving_average_model_3_mv_path = f"{self.experiment_folder_path}/moving_average_model_3_mv.zip"
            if os.path.exists(moving_average_model_3_mv_path):
                self.moving_average_model_3_mv = self.model_type.load(moving_average_model_3_mv_path)
                self.moving_average_model_3_mv.training = False
                self.test_ma_3_mv = self.test_model(self.moving_average_model_3_mv)[0]
                self.vali_ma_3_mv = self.validate_model(self.moving_average_model_3_mv)[0]
            else:
                logging.warning(f"Moving average model 3 MV not found at {moving_average_model_3_mv_path}, skipping evaluation")
                self.moving_average_model_3_mv = None
                self.test_ma_3_mv = None
                self.vali_ma_3_mv = None

        # Load best mean reward model if it exists
        best_mean_reward_model_path = f"{self.experiment_folder_path}/best_mean_reward_model.zip"
        if os.path.exists(best_mean_reward_model_path):
            self.best_mean_reward_model = self.model_type.load(best_mean_reward_model_path)
            self.best_mean_reward_model.training = False
            self.test_bm = self.test_model(self.best_mean_reward_model)[0]
            self.vali_bm = self.validate_model(self.best_mean_reward_model)[0]
        else:
            logging.warning(f"Best mean reward model not found at {best_mean_reward_model_path}, skipping evaluation")
            self.best_mean_reward_model = None
            self.test_bm = None
            self.vali_bm = None
            
        if len(self.multi_validation_wl) > 0:
            best_mean_reward_model_mv_path = f"{self.experiment_folder_path}/best_mean_reward_model_mv.zip"
            if os.path.exists(best_mean_reward_model_mv_path):
                self.best_mean_reward_model_mv = self.model_type.load(best_mean_reward_model_mv_path)
                self.best_mean_reward_model_mv.training = False
                self.test_bm_mv = self.test_model(self.best_mean_reward_model_mv)[0]
                self.vali_bm_mv = self.validate_model(self.best_mean_reward_model_mv)[0]
            else:
                logging.warning(f"Best mean reward model MV not found at {best_mean_reward_model_mv_path}, skipping evaluation")
                self.best_mean_reward_model_mv = None
                self.test_bm_mv = None
                self.vali_bm_mv = None

        # 保存包含labels的workloads为JSON格式
        self._save_workloads_as_json()

        # 创建包含labels的workload文件
        self._create_workload_files_with_labels()

        self._write_report()

        logging.critical(
            (
                f"Finished training of ID {self.id}. Report can be found at "
                f"./{self.experiment_folder_path}/report_ID_{self.id}.txt"
            )
        )

    def _get_wl_budgets_from_model_perfs(self, perfs):
        wl_budgets = []
        for perf in perfs:
            assert perf["evaluated_workload"].budget == perf["available_budget"], "Budget mismatch!"
            wl_budgets.append(perf["evaluated_workload"].budget)
        return wl_budgets

    def start_learning(self):
        self.training_start_time = datetime.datetime.now()

    def set_model(self, model):
        self.model = model

    def finish_learning(self, training_env, moving_average_model_step, best_mean_model_step):
        self.training_end_time = datetime.datetime.now()

        self.moving_average_validation_model_at_step = moving_average_model_step
        self.best_mean_model_step = best_mean_model_step

        self.model.save(f"{self.experiment_folder_path}/final_model")
        training_env.save(f"{self.experiment_folder_path}/vec_normalize.pkl")

        self.evaluated_episodes = 0
        for number_of_resets in training_env.get_attr("number_of_resets"):
            self.evaluated_episodes += number_of_resets

        self.total_steps_taken = 0
        for total_number_of_steps in training_env.get_attr("total_number_of_steps"):
            self.total_steps_taken += total_number_of_steps

        self.cache_hits = 0
        self.cost_requests = 0
        self.costing_time = datetime.timedelta(0)
        for cache_info in training_env.env_method("get_cost_eval_cache_info"):
            self.cache_hits += cache_info[1]
            self.cost_requests += cache_info[0]
            self.costing_time += cache_info[2]
        self.costing_time /= self.config["parallel_environments"]

        self.cache_hit_ratio = self.cache_hits / self.cost_requests * 100

        if self.config["pickle_cost_estimation_caches"]:
            caches = []
            for cache in training_env.env_method("get_cost_eval_cache"):
                caches.append(cache)
            combined_caches = {}
            for cache in caches:
                combined_caches = {**combined_caches, **cache}
            with gzip.open(f"{self.experiment_folder_path}/caches.pickle.gzip", "wb") as handle:
                pickle.dump(combined_caches, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _init_times(self):
        self.start_time = datetime.datetime.now()

        self.end_time = None
        self.training_start_time = None
        self.training_end_time = None

    def __getstate__(self):
        """自定义pickle序列化，排除不可pickle的属性"""
        state = self.__dict__.copy()
        # 移除不可pickle的属性
        unpicklable_attrs = [
            'model_type', 'set_random_seed', 'evaluate_policy', 'DummyVecEnv',
            'VecNormalize', 'sync_envs_normalization'
        ]
        for attr in unpicklable_attrs:
            state.pop(attr, None)
        return state

    def __setstate__(self, state):
        """自定义pickle反序列化，恢复不可pickle的属性"""
        self.__dict__.update(state)
        # 重新设置StableBaselines版本相关的方法
        self._set_sb_version_specific_methods()
        # model_type需要在加载后重新设置

    def _create_experiment_folder(self):
        assert os.path.isdir(
            self.EXPERIMENT_RESULT_PATH
        ), f"Folder for experiment results should exist at: ./{self.EXPERIMENT_RESULT_PATH}"

        self.experiment_folder_path = f"{self.EXPERIMENT_RESULT_PATH}/ID_{self.id}"
        assert os.path.isdir(self.experiment_folder_path) is False, (
            f"Experiment folder already exists at: ./{self.experiment_folder_path} - "
            "terminating here because we don't want to overwrite anything."
        )

        os.mkdir(self.experiment_folder_path)

    def _write_report(self):
        with open(f"{self.experiment_folder_path}/report_ID_{self.id}.txt", "w") as f:
            f.write(f"##### Report for Experiment with ID: {self.id} #####\n")
            f.write(f"Description: {self.config['description']}\n")
            f.write("\n")

            f.write(f"Start:                         {self.start_time}\n")
            f.write(f"End:                           {self.start_time}\n")
            f.write(f"Duration:                      {self.end_time - self.start_time}\n")
            f.write("\n")
            f.write(f"Start Training:                {self.training_start_time}\n")
            f.write(f"End Training:                  {self.training_end_time}\n")
            f.write(f"Duration Training:             {self.training_end_time - self.training_start_time}\n")
            f.write(f"Moving Average model at step:  {self.moving_average_validation_model_at_step}\n")
            f.write(f"Mean reward model at step:     {self.best_mean_model_step}\n")
            # 安全地获取Git哈希，如果失败则使用占位符
            try:
                git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                git_hash = "No commits yet"
            f.write(f"Git Hash:                      {git_hash}\n")
            f.write(f"Number of features:            {self.number_of_features}\n")
            f.write(f"Number of actions:             {self.number_of_actions}\n")
            f.write("\n")
            
            # 添加索引选择信息到报告
            f.write("##### 索引选择信息 #####\n")
            f.write("注意: 详细的索引选择信息已保存到单独的JSON文件中\n")
            f.write("文件命名格式: index_selection_{environment_type}_{timestamp}.json\n")
            f.write("包含信息: 索引选择顺序、存储消耗、成本改进等\n")
            f.write("\n")
            
            if self.config["workload"]["unknown_queries"] > 0:
                f.write(f"Unknown Query Classes {sorted(self.workload_generator.unknown_query_classes)}\n")
                f.write(f"Known Queries: {self.workload_generator.known_query_classes}\n")
                f.write("\n")
            probabilities = len(self.config["workload"]["validation_testing"]["unknown_query_probabilities"])
            for idx, unknown_query_probability in enumerate(
                self.config["workload"]["validation_testing"]["unknown_query_probabilities"]
            ):
                f.write(f"Unknown query probability: {unknown_query_probability}:\n")
                f.write("    Final mean performance test:\n")
                test_fm_perfs, self.performance_test_final_model, self.test_fm_details = self.test_fm[idx]
                vali_fm_perfs, self.performance_vali_final_model, self.vali_fm_details = self.vali_fm[idx]

                # Handle moving average model results (may be None)
                if self.test_ma is not None:
                    _, self.performance_test_moving_average_model, self.test_ma_details = self.test_ma[idx]
                else:
                    self.performance_test_moving_average_model = float('nan')
                    self.test_ma_details = "N/A"
                    
                if self.vali_ma is not None:
                    _, self.performance_vali_moving_average_model, self.vali_ma_details = self.vali_ma[idx]
                else:
                    self.performance_vali_moving_average_model = float('nan')
                    self.vali_ma_details = "N/A"

                # Handle moving average model 3 results (may be None)
                if self.test_ma_3 is not None:
                    _, self.performance_test_moving_average_model_3, self.test_ma_details_3 = self.test_ma_3[idx]
                else:
                    self.performance_test_moving_average_model_3 = float('nan')
                    self.test_ma_details_3 = "N/A"
                    
                if self.vali_ma_3 is not None:
                    _, self.performance_vali_moving_average_model_3, self.vali_ma_details_3 = self.vali_ma_3[idx]
                else:
                    self.performance_vali_moving_average_model_3 = float('nan')
                    self.vali_ma_details_3 = "N/A"

                # Handle best mean reward model results (may be None)
                if self.test_bm is not None:
                    _, self.performance_test_best_mean_reward_model, self.test_bm_details = self.test_bm[idx]
                else:
                    self.performance_test_best_mean_reward_model = float('nan')
                    self.test_bm_details = "N/A"
                    
                if self.vali_bm is not None:
                    _, self.performance_vali_best_mean_reward_model, self.vali_bm_details = self.vali_bm[idx]
                else:
                    self.performance_vali_best_mean_reward_model = float('nan')
                    self.vali_bm_details = "N/A"

                if len(self.multi_validation_wl) > 0:
                    if self.test_ma_mv is not None:
                        _, self.performance_test_moving_average_model_mv, self.test_ma_details_mv = self.test_ma_mv[idx]
                    else:
                        self.performance_test_moving_average_model_mv = float('nan')
                        self.test_ma_details_mv = "N/A"
                        
                    if self.vali_ma_mv is not None:
                        _, self.performance_vali_moving_average_model_mv, self.vali_ma_details_mv = self.vali_ma_mv[idx]
                    else:
                        self.performance_vali_moving_average_model_mv = float('nan')
                        self.vali_ma_details_mv = "N/A"
                        
                    if self.test_ma_3_mv is not None:
                        _, self.performance_test_moving_average_model_3_mv, self.test_ma_details_3_mv = self.test_ma_3_mv[idx]
                    else:
                        self.performance_test_moving_average_model_3_mv = float('nan')
                        self.test_ma_details_3_mv = "N/A"
                        
                    if self.vali_ma_3_mv is not None:
                        _, self.performance_vali_moving_average_model_3_mv, self.vali_ma_details_3_mv = self.vali_ma_3_mv[idx]
                    else:
                        self.performance_vali_moving_average_model_3_mv = float('nan')
                        self.vali_ma_details_3_mv = "N/A"
                        
                    if self.test_bm_mv is not None:
                        _, self.performance_test_best_mean_reward_model_mv, self.test_bm_details_mv = self.test_bm_mv[idx]
                    else:
                        self.performance_test_best_mean_reward_model_mv = float('nan')
                        self.test_bm_details_mv = "N/A"
                        
                    if self.vali_bm_mv is not None:
                        _, self.performance_vali_best_mean_reward_model_mv, self.vali_bm_details_mv = self.vali_bm_mv[idx]
                    else:
                        self.performance_vali_best_mean_reward_model_mv = float('nan')
                        self.vali_bm_details_mv = "N/A"

                self.test_fm_wl_budgets = self._get_wl_budgets_from_model_perfs(test_fm_perfs)
                self.vali_fm_wl_budgets = self._get_wl_budgets_from_model_perfs(vali_fm_perfs)

                f.write(
                    (
                        "        Final model:               "
                        f"{self.performance_test_final_model:.2f} ({self.test_fm_details})\n"
                    )
                )
                # Handle NaN values in moving average model
                if not np.isnan(self.performance_test_moving_average_model):
                    f.write(
                        (
                            "        Moving Average model:      "
                            f"{self.performance_test_moving_average_model:.2f} ({self.test_ma_details})\n"
                        )
                    )
                else:
                    f.write("        Moving Average model:      N/A\n")
                    
                if len(self.multi_validation_wl) > 0:
                    if not np.isnan(self.performance_test_moving_average_model_mv):
                        f.write(
                            (
                                "        Moving Average model (MV): "
                                f"{self.performance_test_moving_average_model_mv:.2f} ({self.test_ma_details_mv})\n"
                            )
                        )
                    else:
                        f.write("        Moving Average model (MV): N/A\n")
                        
                # Handle NaN values in moving average 3 model
                if not np.isnan(self.performance_test_moving_average_model_3):
                    f.write(
                        (
                            "        Moving Average 3 model:    "
                            f"{self.performance_test_moving_average_model_3:.2f} ({self.test_ma_details_3})\n"
                        )
                    )
                else:
                    f.write("        Moving Average 3 model:    N/A\n")
                    
                if len(self.multi_validation_wl) > 0:
                    if not np.isnan(self.performance_test_moving_average_model_3_mv):
                        f.write(
                            (
                                "        Moving Average 3 mod (MV): "
                                f"{self.performance_test_moving_average_model_3_mv:.2f} ({self.test_ma_details_3_mv})\n"
                            )
                        )
                    else:
                        f.write("        Moving Average 3 mod (MV): N/A\n")
                        
                # Handle NaN values in best mean reward model
                if not np.isnan(self.performance_test_best_mean_reward_model):
                    f.write(
                        (
                            "        Best mean reward model:    "
                            f"{self.performance_test_best_mean_reward_model:.2f} ({self.test_bm_details})\n"
                        )
                    )
                else:
                    f.write("        Best mean reward model:    N/A\n")
                    
                if len(self.multi_validation_wl) > 0:
                    if not np.isnan(self.performance_test_best_mean_reward_model_mv):
                        f.write(
                            (
                                "        Best mean reward mod (MV): "
                                f"{self.performance_test_best_mean_reward_model_mv:.2f} ({self.test_bm_details_mv})\n"
                            )
                        )
                    else:
                        f.write("        Best mean reward mod (MV): N/A\n")
                for key, value in self.comparison_performances["test"].items():
                    if len(value) < 1:
                        continue
                    f.write(f"        {key}:                    {np.mean(value[idx]):.2f} ({value[idx]})\n")
                f.write("\n")
                f.write(f"        Budgets:                   {self.test_fm_wl_budgets}\n")
                f.write("\n")
                f.write("    Final mean performance validation:\n")
                f.write(
                    (
                        "        Final model:               "
                        f"{self.performance_vali_final_model:.2f} ({self.vali_fm_details})\n"
                    )
                )
                # Handle NaN values in validation moving average model
                if not np.isnan(self.performance_vali_moving_average_model):
                    f.write(
                        (
                            "        Moving Average model:      "
                            f"{self.performance_vali_moving_average_model:.2f} ({self.vali_ma_details})\n"
                        )
                    )
                else:
                    f.write("        Moving Average model:      N/A\n")
                    
                if len(self.multi_validation_wl) > 0:
                    if not np.isnan(self.performance_vali_moving_average_model_mv):
                        f.write(
                            (
                                "        Moving Average model (MV): "
                                f"{self.performance_vali_moving_average_model_mv:.2f} ({self.vali_ma_details_mv})\n"
                            )
                        )
                    else:
                        f.write("        Moving Average model (MV): N/A\n")
                # Handle NaN values in validation moving average 3 model
                if not np.isnan(self.performance_vali_moving_average_model_3):
                    f.write(
                        (
                            "        Moving Average 3 model:    "
                            f"{self.performance_vali_moving_average_model_3:.2f} ({self.vali_ma_details_3})\n"
                        )
                    )
                else:
                    f.write("        Moving Average 3 model:    N/A\n")
                    
                if len(self.multi_validation_wl) > 0:
                    if not np.isnan(self.performance_vali_moving_average_model_3_mv):
                        f.write(
                            (
                                "        Moving Average 3 mod (MV): "
                                f"{self.performance_vali_moving_average_model_3_mv:.2f} ({self.vali_ma_details_3_mv})\n"
                            )
                        )
                    else:
                        f.write("        Moving Average 3 mod (MV): N/A\n")
                        
                # Handle NaN values in validation best mean reward model
                if not np.isnan(self.performance_vali_best_mean_reward_model):
                    f.write(
                        (
                            "        Best mean reward model:    "
                            f"{self.performance_vali_best_mean_reward_model:.2f} ({self.vali_bm_details})\n"
                        )
                    )
                else:
                    f.write("        Best mean reward model:    N/A\n")
                    
                if len(self.multi_validation_wl) > 0:
                    if not np.isnan(self.performance_vali_best_mean_reward_model_mv):
                        f.write(
                            (
                                "        Best mean reward mod (MV): "
                                f"{self.performance_vali_best_mean_reward_model_mv:.2f} ({self.vali_bm_details_mv})\n"
                            )
                        )
                    else:
                        f.write("        Best mean reward mod (MV): N/A\n")
                for key, value in self.comparison_performances["validation"].items():
                    if len(value) < 1:
                        continue
                    f.write(f"        {key}:                    {np.mean(value[idx]):.2f} ({value[idx]})\n")
                f.write("\n")
                f.write(f"        Budgets:                   {self.vali_fm_wl_budgets}\n")
                f.write("\n")
                f.write("\n")
            f.write("Overall Test:\n")

            def final_avg(values, probabilities):
                val = 0
                for res in values:
                    val += res[1]
                return val / probabilities

            f.write(("        Final model:               " f"{final_avg(self.test_fm, probabilities):.2f}\n"))
            
            # Handle None values for moving average models
            if self.test_ma is not None:
                f.write(("        Moving Average model:      " f"{final_avg(self.test_ma, probabilities):.2f}\n"))
            else:
                f.write("        Moving Average model:      N/A\n")
                
            if len(self.multi_validation_wl) > 0:
                if self.test_ma_mv is not None:
                    f.write(("        Moving Average model (MV): " f"{final_avg(self.test_ma_mv, probabilities):.2f}\n"))
                else:
                    f.write("        Moving Average model (MV): N/A\n")
                    
            if self.test_ma_3 is not None:
                f.write(("        Moving Average 3 model:    " f"{final_avg(self.test_ma_3, probabilities):.2f}\n"))
            else:
                f.write("        Moving Average 3 model:    N/A\n")
                
            if len(self.multi_validation_wl) > 0:
                if self.test_ma_3_mv is not None:
                    f.write(("        Moving Average 3 mod (MV): " f"{final_avg(self.test_ma_3_mv, probabilities):.2f}\n"))
                else:
                    f.write("        Moving Average 3 mod (MV): N/A\n")
                    
            if self.test_bm is not None:
                f.write(("        Best mean reward model:    " f"{final_avg(self.test_bm, probabilities):.2f}\n"))
            else:
                f.write("        Best mean reward model:    N/A\n")
                
            if len(self.multi_validation_wl) > 0:
                if self.test_bm_mv is not None:
                    f.write(("        Best mean reward mod (MV): " f"{final_avg(self.test_bm_mv, probabilities):.2f}\n"))
                else:
                    f.write("        Best mean reward mod (MV): N/A\n")
            f.write(
                (
                    "        Extend:                    "
                    f"{np.mean(self.comparison_performances['test']['Extend']):.2f}\n"
                )
            )
            f.write(
                (
                    "        DB2Adv:                    "
                    f"{np.mean(self.comparison_performances['test']['DB2Adv']):.2f}\n"
                )
            )
            f.write("\n")
            
            # 添加时间统计到报告
            f.write("##### 时间统计信息 #####\n")
            f.write("索引选择算法执行时间对比:\n")
            f.write("\n")
            
            # Test时间统计
            f.write("Test环境时间统计:\n")
            if hasattr(self, 'swirl_times') and 'test' in self.swirl_times and len(self.swirl_times['test']) > 0:
                swirl_avg_time = np.mean(self.swirl_times['test'])
                f.write(f"    SWIRL平均时间:              {swirl_avg_time:.4f} 秒\n")
            if hasattr(self, 'extend_times') and 'test' in self.extend_times and len(self.extend_times['test']) > 0:
                extend_avg_time = np.mean(self.extend_times['test'])
                f.write(f"    Extend平均时间:             {extend_avg_time:.4f} 秒\n")
            if hasattr(self, 'db2advis_times') and 'test' in self.db2advis_times and len(self.db2advis_times['test']) > 0:
                db2advis_avg_time = np.mean(self.db2advis_times['test'])
                f.write(f"    DB2Advis平均时间:           {db2advis_avg_time:.4f} 秒\n")
            
            # Validation时间统计
            f.write("Validation环境时间统计:\n")
            if hasattr(self, 'swirl_times') and 'validation' in self.swirl_times and len(self.swirl_times['validation']) > 0:
                swirl_avg_time = np.mean(self.swirl_times['validation'])
                f.write(f"    SWIRL平均时间:              {swirl_avg_time:.4f} 秒\n")
            if hasattr(self, 'extend_times') and 'validation' in self.extend_times and len(self.extend_times['validation']) > 0:
                extend_avg_time = np.mean(self.extend_times['validation'])
                f.write(f"    Extend平均时间:             {extend_avg_time:.4f} 秒\n")
            if hasattr(self, 'db2advis_times') and 'validation' in self.db2advis_times and len(self.db2advis_times['validation']) > 0:
                db2advis_avg_time = np.mean(self.db2advis_times['validation'])
                f.write(f"    DB2Advis平均时间:           {db2advis_avg_time:.4f} 秒\n")
            
            f.write("\n")
            f.write("Overall Validation:\n")
            f.write(("        Final model:               " f"{final_avg(self.vali_fm, probabilities):.2f}\n"))
            
            # Handle None values for validation moving average models
            if self.vali_ma is not None:
                f.write(("        Moving Average model:      " f"{final_avg(self.vali_ma, probabilities):.2f}\n"))
            else:
                f.write("        Moving Average model:      N/A\n")
                
            if len(self.multi_validation_wl) > 0:
                if self.vali_ma_mv is not None:
                    f.write(("        Moving Average model (MV): " f"{final_avg(self.vali_ma_mv, probabilities):.2f}\n"))
                else:
                    f.write("        Moving Average model (MV): N/A\n")
                    
            if self.vali_ma_3 is not None:
                f.write(("        Moving Average 3 model:    " f"{final_avg(self.vali_ma_3, probabilities):.2f}\n"))
            else:
                f.write("        Moving Average 3 model:    N/A\n")
                
            if len(self.multi_validation_wl) > 0:
                if self.vali_ma_3_mv is not None:
                    f.write(("        Moving Average 3 mod (MV): " f"{final_avg(self.vali_ma_3_mv, probabilities):.2f}\n"))
                else:
                    f.write("        Moving Average 3 mod (MV): N/A\n")
                    
            if self.vali_bm is not None:
                f.write(("        Best mean reward model:    " f"{final_avg(self.vali_bm, probabilities):.2f}\n"))
            else:
                f.write("        Best mean reward model:    N/A\n")
                
            if len(self.multi_validation_wl) > 0:
                if self.vali_bm_mv is not None:
                    f.write(("        Best mean reward mod (MV): " f"{final_avg(self.vali_bm_mv, probabilities):.2f}\n"))
                else:
                    f.write("        Best mean reward mod (MV): N/A\n")
            f.write(
                (
                    "        Extend:                    "
                    f"{np.mean(self.comparison_performances['validation']['Extend']):.2f}\n"
                )
            )
            f.write(
                (
                    "        DB2Adv:                    "
                    f"{np.mean(self.comparison_performances['validation']['DB2Adv']):.2f}\n"
                )
            )
            f.write("\n")
            f.write("\n")
            
            # 添加总体时间统计
            f.write("##### 总体时间统计 #####\n")
            total_swirl_time = 0.0
            total_extend_time = 0.0
            total_db2advis_time = 0.0
            count_swirl = 0
            count_extend = 0
            count_db2advis = 0
            
            if hasattr(self, 'swirl_times'):
                for env_type in ['test', 'validation']:
                    if env_type in self.swirl_times and len(self.swirl_times[env_type]) > 0:
                        total_swirl_time += sum(self.swirl_times[env_type])
                        count_swirl += len(self.swirl_times[env_type])
            
            if hasattr(self, 'extend_times'):
                for env_type in ['test', 'validation']:
                    if env_type in self.extend_times and len(self.extend_times[env_type]) > 0:
                        total_extend_time += sum(self.extend_times[env_type])
                        count_extend += len(self.extend_times[env_type])
            
            if hasattr(self, 'db2advis_times'):
                for env_type in ['test', 'validation']:
                    if env_type in self.db2advis_times and len(self.db2advis_times[env_type]) > 0:
                        total_db2advis_time += sum(self.db2advis_times[env_type])
                        count_db2advis += len(self.db2advis_times[env_type])
            
            if count_swirl > 0:
                f.write(f"SWIRL总体平均时间:            {total_swirl_time/count_swirl:.4f} 秒 (共{count_swirl}个workload)\n")
            if count_extend > 0:
                f.write(f"Extend总体平均时间:           {total_extend_time/count_extend:.4f} 秒 (共{count_extend}个workload)\n")
            if count_db2advis > 0:
                f.write(f"DB2Advis总体平均时间:         {total_db2advis_time/count_db2advis:.4f} 秒 (共{count_db2advis}个workload)\n")
            
            f.write("\n")
            f.write(f"Evaluated episodes:            {self.evaluated_episodes}\n")
            f.write(f"Total steps taken:             {self.total_steps_taken}\n")
            f.write(
                (
                    f"CostEval cache hit ratio:      "
                    f"{self.cache_hit_ratio:.2f} ({self.cache_hits} of {self.cost_requests})\n"
                )
            )
            training_time = self.training_end_time - self.training_start_time
            f.write(
                f"Cost eval time (% of total):   {self.costing_time} ({self.costing_time / training_time * 100:.2f}%)\n"
            )
            # f.write(f"Cost eval time:                {self.costing_time:.2f}\n")

            f.write("\n\n")
            f.write("Used configuration:\n")
            json.dump(self.config, f)
            f.write("\n\n")
            f.write("Evaluated test workloads:\n")
            for evaluated_workload in self.evaluated_workloads_strs[: (len(self.evaluated_workloads_strs) // 2)]:
                f.write(f"{evaluated_workload}\n")
            f.write("Evaluated validation workloads:\n")
            # fmt: off
            for evaluated_workload in self.evaluated_workloads_strs[(len(self.evaluated_workloads_strs) // 2) :]:  # noqa: E203, E501
                f.write(f"{evaluated_workload}\n")
            # fmt: on
            f.write("\n\n")

    def compare(self):
        if len(self.config["comparison_algorithms"]) < 1:
            return

        if "extend" in self.config["comparison_algorithms"]:
            self._compare_extend()
        if "db2advis" in self.config["comparison_algorithms"]:
            self._compare_db2advis()
        for key, comparison_performance in self.comparison_performances.items():
            print(f"Comparison for {key}:")
            for key, value in comparison_performance.items():
                print(f"    {key}: {np.mean(value):.2f} ({value})")
            
            # 添加时间统计
            print(f"Time statistics for {key}:")
            if hasattr(self, 'swirl_times') and key in self.swirl_times:
                swirl_avg_time = np.mean(self.swirl_times[key])
                print(f"    SWIRL average time: {swirl_avg_time:.4f} seconds")
            if hasattr(self, 'extend_times') and key in self.extend_times:
                extend_avg_time = np.mean(self.extend_times[key])
                print(f"    Extend average time: {extend_avg_time:.4f} seconds")
            if hasattr(self, 'db2advis_times') and key in self.db2advis_times:
                db2advis_avg_time = np.mean(self.db2advis_times[key])
                print(f"    DB2Advis average time: {db2advis_avg_time:.4f} seconds")

        self._evaluate_comparison()

    def _evaluate_comparison(self):
        for key, comparison_indexes in self.comparison_indexes.items():
            columns_from_indexes = set()
            for index in comparison_indexes:
                for column in index.columns:
                    columns_from_indexes |= set([column])

            impossible_index_columns = columns_from_indexes - self.single_column_flat_set
            logging.critical(f"{key} finds indexes on these not indexable columns:\n    {impossible_index_columns}")

            assert len(impossible_index_columns) == 0, "Found indexes on not indexable columns."

    def _compare_extend(self):
        self.evaluated_workloads = set()
        for model_performances_outer, run_type in [self.test_model(self.model), self.validate_model(self.model)]:
            for model_performances, _, _ in model_performances_outer:
                self.comparison_performances[run_type]["Extend"].append([])
                for model_performance in model_performances:
                    assert (
                        model_performance["evaluated_workload"].budget == model_performance["available_budget"]
                    ), "Budget mismatch!"
                    assert model_performance["evaluated_workload"] not in self.evaluated_workloads
                    self.evaluated_workloads.add(model_performance["evaluated_workload"])

                    parameters = {
                        "budget_MB": model_performance["evaluated_workload"].budget,
                        "max_index_width": self.config["max_index_width"],
                        "min_cost_improvement": 1.003,
                    }
                    # Get database connection parameters from environment variables
                    db_host = os.getenv('DATABASE_HOST', 'localhost')
                    db_port = os.getenv('DATABASE_PORT', '54321')
                    extend_connector = PostgresDatabaseConnector(self.schema.database_name, autocommit=True, host=db_host, port=db_port)
                    extend_connector.drop_indexes()
                    extend_algorithm = ExtendAlgorithm(extend_connector, parameters)
                    indexes = extend_algorithm.calculate_best_indexes(model_performance["evaluated_workload"])
                    self.comparison_indexes["Extend"] |= frozenset(indexes)

                    # 记录Extend算法的时间
                    extend_time = getattr(extend_algorithm, 'index_selection_time', 0.0)
                    logging.warning(f"Extend algorithm for workload {model_performance['evaluated_workload']} took {extend_time:.4f} seconds")
                    
                    # 收集Extend算法的时间数据
                    self.extend_times[run_type].append(extend_time)

                    self.comparison_performances[run_type]["Extend"][-1].append(extend_algorithm.final_cost_proportion)

    def _compare_db2advis(self):
        for model_performances_outer, run_type in [self.test_model(self.model), self.validate_model(self.model)]:
            for model_performances, _, _ in model_performances_outer:
                self.comparison_performances[run_type]["DB2Adv"].append([])
                for model_performance in model_performances:
                    parameters = {
                        "budget_MB": model_performance["available_budget"],
                        "max_index_width": self.config["max_index_width"],
                        "try_variations_seconds": 0,
                        # ensure DB2Advis only proposes indexes on allowed columns
                        "allowed_single_columns": self.single_column_flat_set,
                    }
                    # Get database connection parameters from environment variables
                    db_host = os.getenv('DATABASE_HOST', 'localhost')
                    db_port = os.getenv('DATABASE_PORT', '54321')
                    db2advis_connector = PostgresDatabaseConnector(self.schema.database_name, autocommit=True, host=db_host, port=db_port)
                    db2advis_connector.drop_indexes()
                    db2advis_algorithm = DB2AdvisAlgorithm(db2advis_connector, parameters)
                    indexes = db2advis_algorithm.calculate_best_indexes(model_performance["evaluated_workload"])
                    self.comparison_indexes["DB2Adv"] |= frozenset(indexes)

                    # 记录DB2Advis算法的时间
                    db2advis_time = getattr(db2advis_algorithm, 'index_selection_time', 0.0)
                    logging.warning(f"DB2Advis algorithm for workload {model_performance['evaluated_workload']} took {db2advis_time:.4f} seconds")
                    
                    # 收集DB2Advis算法的时间数据
                    self.db2advis_times[run_type].append(db2advis_time)

                    self.comparison_performances[run_type]["DB2Adv"][-1].append(
                        db2advis_algorithm.final_cost_proportion
                    )

                    self.evaluated_workloads_strs.append(f"{model_performance['evaluated_workload']}\n")

    # todo: code duplication with validate_model
    def test_model(self, model):
        model_performances = []
        for test_wl in self.workload_generator.wl_testing:
            test_env = self.DummyVecEnv([self.make_env(0, EnvironmentType.TESTING, test_wl)])
            test_env = self.VecNormalize(
                test_env, norm_obs=True, norm_reward=False, gamma=self.config["rl_algorithm"]["gamma"], training=False
            )

            if model != self.model:
                model.set_env(self.model.env)

            model_performance = self._evaluate_model(model, test_env, len(test_wl))
            model_performances.append(model_performance)

        return model_performances, "test"

    def validate_model(self, model):
        model_performances = []
        for validation_wl in self.workload_generator.wl_validation:
            validation_env = self.DummyVecEnv([self.make_env(0, EnvironmentType.VALIDATION, validation_wl)])
            validation_env = self.VecNormalize(
                validation_env,
                norm_obs=True,
                norm_reward=False,
                gamma=self.config["rl_algorithm"]["gamma"],
                training=False,
            )

            if model != self.model:
                model.set_env(self.model.env)

            model_performance = self._evaluate_model(model, validation_env, len(validation_wl))
            model_performances.append(model_performance)

        return model_performances, "validation"

    import time

    def _evaluate_model(self, model, evaluation_env, n_eval_episodes):
        training_env = model.get_vec_normalize_env()
        self.sync_envs_normalization(training_env, evaluation_env)

        # 记录SWIRL决策时间
        start_time = time.time()
        self.evaluate_policy(model, evaluation_env, n_eval_episodes)
        end_time = time.time()
        swirl_selection_time = end_time - start_time

        episode_performances = evaluation_env.get_attr("episode_performances")[0]
        perfs = []
        for perf in episode_performances:
            perfs.append(round(perf["achieved_cost"], 2))
            # 这里可以为每个episode加上swirl_selection_time
            perf["swirl_selection_time"] = swirl_selection_time

        mean_performance = np.mean(perfs)
        print(f"Mean performance: {mean_performance:.2f} ({perfs})")

        # 收集推荐索引作为labels（不再生成JSON文件）
        self._collect_index_labels(episode_performances, evaluation_env)

        # 统计到self.swirl_times
        env_type = evaluation_env.get_attr("environment_type")[0]
        if env_type == EnvironmentType.TESTING:
            run_type = "test"
        elif env_type == EnvironmentType.VALIDATION:
            run_type = "validation"
        else:
            run_type = "unknown"
        if not hasattr(self, "swirl_times"):
            self.swirl_times = {"test": [], "validation": []}
        self.swirl_times[run_type].append(swirl_selection_time)

        return episode_performances, mean_performance, perfs
    def _output_index_selection_info(self, episode_performances, evaluation_env):
        """输出索引选择信息到文件"""
        import json
        from datetime import datetime
        
        # 获取环境类型
        env_type = evaluation_env.get_attr("environment_type")[0]
        
        # 创建索引输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        index_output_file = f"{self.experiment_folder_path}/index_selection_{env_type}_{timestamp}.json"
        
        # 收集索引选择信息
        index_selection_data = {
            "experiment_id": self.id,
            "environment_type": str(env_type),
            "timestamp": timestamp,
            "episodes": [],
            "time_statistics": {
                "total_episodes": len(episode_performances),
                "average_selection_time": 0.0,
                "total_selection_time": 0.0
            }
        }
        
        for i, episode_perf in enumerate(episode_performances):
            episode_data = {
                "episode_id": i,
                "workload": str(episode_perf["evaluated_workload"]),
                "budget_mb": episode_perf["available_budget"],
                "initial_cost": episode_perf.get("initial_cost", 0),
                "final_cost": episode_perf.get("current_cost", 0),
                "cost_improvement_percent": episode_perf["achieved_cost"],
                "storage_consumption_mb": episode_perf["memory_consumption"] / (1024 * 1024),
                "final_indexes": [str(idx) for idx in episode_perf["indexes"]],
                "index_selection_sequence": [],
                "index_selection_time": episode_perf.get("index_selection_time", 0.0)
            }
            
            # 添加索引选择顺序信息
            if "selected_indexes_sequence" in episode_perf:
                for seq_item in episode_perf["selected_indexes_sequence"]:
                    episode_data["index_selection_sequence"].append({
                        "step": seq_item["step"],
                        "action": seq_item["action"],
                        "index": str(seq_item["index"]),
                        "index_size_mb": seq_item["index_size"] / (1024 * 1024),
                        "current_storage_mb": seq_item["current_storage"] / (1024 * 1024)
                    })
            
            index_selection_data["episodes"].append(episode_data)
            
            # 更新时间统计
            index_selection_data["time_statistics"]["total_selection_time"] += episode_data["index_selection_time"]
        
        # 计算平均时间
        if index_selection_data["time_statistics"]["total_episodes"] > 0:
            index_selection_data["time_statistics"]["average_selection_time"] = (
                index_selection_data["time_statistics"]["total_selection_time"] / 
                index_selection_data["time_statistics"]["total_episodes"]
            )
        
        # 保存到JSON文件
        with open(index_output_file, 'w', encoding='utf-8') as f:
            json.dump(index_selection_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        # 同时输出到控制台
        logging.info(f"Index selection information saved to: {index_output_file}")
        
        # 输出摘要信息
        self._print_index_selection_summary(index_selection_data)
    
    def _print_index_selection_summary(self, index_selection_data):
        """打印索引选择摘要信息"""
        print(f"\n=== 索引选择摘要 ({index_selection_data['environment_type']}) ===")
        print(f"实验ID: {index_selection_data['experiment_id']}")
        print(f"时间戳: {index_selection_data['timestamp']}")
        print(f"总工作负载数: {len(index_selection_data['episodes'])}")
        
        total_indexes = 0
        total_storage = 0
        avg_cost_improvement = 0
        
        for episode in index_selection_data["episodes"]:
            total_indexes += len(episode["final_indexes"])
            total_storage += episode["storage_consumption_mb"]
            avg_cost_improvement += episode["cost_improvement_percent"]
        
        if index_selection_data["episodes"]:
            avg_cost_improvement /= len(index_selection_data["episodes"])
            avg_indexes = total_indexes / len(index_selection_data["episodes"])
            avg_storage = total_storage / len(index_selection_data["episodes"])
            
            print(f"平均索引数量: {avg_indexes:.2f}")
            print(f"平均存储消耗: {avg_storage:.2f} MB")
            print(f"平均成本改进: {avg_cost_improvement:.2f}%")
        
        print("=" * 50)

    def _collect_index_labels(self, episode_performances, evaluation_env):
        """收集推荐索引作为workload的labels"""
        env_type = evaluation_env.get_attr("environment_type")[0]
        run_type = "test" if env_type == EnvironmentType.TESTING else "validation"

        print(f"🔍 DEBUG: 开始收集 {run_type} 的workload labels，共 {len(episode_performances)} 个episodes")
        logging.info(f"开始收集 {run_type} 的workload labels，共 {len(episode_performances)} 个episodes")

        # 记录实验ID和时间戳
        print(f"🔍 DEBUG: 实验ID: {self.id}, 当前时间: {datetime.datetime.now()}")
        logging.info(f"实验ID: {self.id}, 当前时间: {datetime.datetime.now()}")

        for i, episode_perf in enumerate(episode_performances):
            workload = episode_perf["evaluated_workload"]

            # 调试：检查episode_perf结构
            logging.info(f"Episode {i}: 检查数据结构")
            logging.info(f"  - episode_perf keys: {list(episode_perf.keys())}")
            logging.info(f"  - workload description: {workload.description}")
            logging.info(f"  - workload id: {getattr(workload, 'id', 'N/A')}")
            logging.info(f"  - workload db: {getattr(workload, 'db', 'N/A')}")

            # 检查indexes字段是否存在
            if "indexes" in episode_perf:
                recommended_indexes = set(episode_perf["indexes"])
                print(f"🔍 DEBUG: Episode {i} 发现 {len(recommended_indexes)} 个推荐索引")
                logging.info(f"  - 发现 {len(recommended_indexes)} 个推荐索引")

                # 显示前几个索引
                for j, idx in enumerate(list(recommended_indexes)[:3]):
                    logging.info(f"    索引 {j+1}: {idx}")
                if len(recommended_indexes) > 3:
                    logging.info(f"    ... 还有 {len(recommended_indexes) - 3} 个索引")
            else:
                recommended_indexes = set()
                logging.warning(f"  - ❌ episode_perf中没有'indexes'字段！")
                logging.warning(f"    可用的字段: {list(episode_perf.keys())}")

            # 将索引添加到workload的labels中
            workload.labels = recommended_indexes
            logging.info(f"  - 已将 {len(recommended_indexes)} 个索引添加到workload.labels")

            # 存储到workload_labels字典中
            workload_key = f"{workload.description}_{i}"
            self.workload_labels[run_type][workload_key] = {
                "workload": workload,
                "recommended_indexes": recommended_indexes,
                "cost_improvement": float(episode_perf["achieved_cost"])
            }

            print(f"🔍 DEBUG: Episode {i} 完成 - workload: {workload.description}, labels: {len(workload.labels)}, key: {workload_key}")
            logging.info(f"Episode {i} 完成 - workload: {workload.description}, labels: {len(workload.labels)}, key: {workload_key}")

        print(f"🔍 DEBUG: ✅ 完成收集 {run_type} 的workload labels，共处理 {len(episode_performances)} 个episodes")
        print(f"🔍 DEBUG:    {run_type} workload_labels 中现在有 {len(self.workload_labels[run_type])} 个条目")
        logging.info(f"✅ 完成收集 {run_type} 的workload labels，共处理 {len(episode_performances)} 个episodes")
        logging.info(f"   {run_type} workload_labels 中现在有 {len(self.workload_labels[run_type])} 个条目")

    def make_env(self, env_id, environment_type=EnvironmentType.TRAINING, workloads_in=None):
        def _init():
            action_manager_class = getattr(
                importlib.import_module("swirl.action_manager"), self.config["action_manager"]
            )
            action_manager = action_manager_class(
                indexable_column_combinations=self.globally_indexable_columns,
                action_storage_consumptions=self.action_storage_consumptions,
                sb_version=self.config["rl_algorithm"]["stable_baselines_version"],
                max_index_width=self.config["max_index_width"],
                reenable_indexes=self.config["reenable_indexes"],
            )

            if self.number_of_actions is None:
                self.number_of_actions = action_manager.number_of_actions

            observation_manager_config = {
                "number_of_query_classes": self.workload_generator.number_of_query_classes,
                "workload_embedder": self.workload_embedder if "workload_embedder" in self.config else None,
                "workload_size": self.config["workload"]["size"],
            }
            observation_manager_class = getattr(
                importlib.import_module("swirl.observation_manager"), self.config["observation_manager"]
            )
            observation_manager = observation_manager_class(
                action_manager.number_of_columns, observation_manager_config
            )

            if self.number_of_features is None:
                self.number_of_features = observation_manager.number_of_features

            reward_calculator_class = getattr(
                importlib.import_module("swirl.reward_calculator"), self.config["reward_calculator"]
            )
            reward_calculator = reward_calculator_class()

            if environment_type == EnvironmentType.TRAINING:
                workloads = self.workload_generator.wl_training if workloads_in is None else workloads_in
            elif environment_type == EnvironmentType.TESTING:
                # Selecting the hardest workload by default
                workloads = self.workload_generator.wl_testing[-1] if workloads_in is None else workloads_in
            elif environment_type == EnvironmentType.VALIDATION:
                # Selecting the hardest workload by default
                workloads = self.workload_generator.wl_validation[-1] if workloads_in is None else workloads_in
            else:
                raise ValueError

            env = gym.make(
                f"DB-v{self.config['gym_version']}",
                environment_type=environment_type,
                config={
                    "database_name": self.schema.database_name,
                    "globally_indexable_columns": self.globally_indexable_columns_flat,
                    "workloads": workloads,
                    "random_seed": self.config["random_seed"] + env_id,
                    "max_steps_per_episode": self.config["max_steps_per_episode"],
                    "action_manager": action_manager,
                    "observation_manager": observation_manager,
                    "reward_calculator": reward_calculator,
                    "env_id": env_id,
                    "similar_workloads": self.config["workload"]["similar_workloads"],
                },
            )
            return env

        self.set_random_seed(self.config["random_seed"])

        return _init

    def _set_sb_version_specific_methods(self):
        if self.config["rl_algorithm"]["stable_baselines_version"] == 2:
            from stable_baselines.common import set_global_seeds as set_global_seeds_sb2
            from stable_baselines.common.evaluation import evaluate_policy as evaluate_policy_sb2
            from stable_baselines.common.vec_env import DummyVecEnv as DummyVecEnv_sb2
            from stable_baselines.common.vec_env import VecNormalize as VecNormalize_sb2
            from stable_baselines.common.vec_env import sync_envs_normalization as sync_envs_normalization_sb2

            self.set_random_seed = set_global_seeds_sb2
            self.evaluate_policy = evaluate_policy_sb2
            self.DummyVecEnv = DummyVecEnv_sb2
            self.VecNormalize = VecNormalize_sb2
            self.sync_envs_normalization = sync_envs_normalization_sb2
        elif self.config["rl_algorithm"]["stable_baselines_version"] == 3:
            raise ValueError("Currently, only StableBaselines 2 is supported.")

            from stable_baselines3.common.evaluation import evaluate_policy as evaluate_policy_sb3
            from stable_baselines3.common.utils import set_random_seed as set_random_seed_sb3
            from stable_baselines3.common.vec_env import DummyVecEnv as DummyVecEnv_sb3
            from stable_baselines3.common.vec_env import VecNormalize as VecNormalize_sb3
            from stable_baselines3.common.vec_env import sync_envs_normalization as sync_envs_normalization_sb3

            self.set_random_seed = set_random_seed_sb3
            self.evaluate_policy = evaluate_policy_sb3
            self.DummyVecEnv = DummyVecEnv_sb3
            self.VecNormalize = VecNormalize_sb3
            self.sync_envs_normalization = sync_envs_normalization_sb3
        else:
            raise ValueError("There are only versions 2 and 3 of StableBaselines.")

    def _create_workload_files_with_labels(self):
        """创建包含推荐索引labels的workload文件"""
        import json
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 处理validation workloads
        if self.workload_labels["validation"]:
            validation_output = []
            for workload_key, data in self.workload_labels["validation"].items():
                workload = data["workload"]
                recommended_indexes = data["recommended_indexes"]

                # 创建包含labels的workload数据结构
                workload_data = {
                    "db": workload.db if workload.db is not None else 'unknown',
                    "id": workload.id if workload.id is not None else 0,
                    "description": workload.description,
                    "queries": [
                        {
                            "sql": query.text,
                            "frequency": int(query.frequency),
                            "tables_columns": self._extract_tables_columns(query)
                        } for query in workload.queries
                    ],
                    "labels": [
                        {
                            "table": index.table().name,
                            "columns": [col.name for col in index.columns]
                        } for index in recommended_indexes
                    ],
                    "cost_improvement": data["cost_improvement"]
                }
                validation_output.append(workload_data)

            # 保存validation workload文件
            validation_file = f"{self.experiment_folder_path}/validation_workloads_with_labels_{timestamp}.json"
            with open(validation_file, 'w', encoding='utf-8') as f:
                json.dump(validation_output, f, indent=2, ensure_ascii=False)
            logging.info(f"Validation workloads with labels saved to: {validation_file}")

        # 处理test workloads
        if self.workload_labels["test"]:
            test_output = []
            for workload_key, data in self.workload_labels["test"].items():
                workload = data["workload"]
                recommended_indexes = data["recommended_indexes"]

                # 创建包含labels的workload数据结构
                workload_data = {
                    "db": workload.db if workload.db is not None else 'unknown',
                    "id": workload.id if workload.id is not None else 0,
                    "description": workload.description,
                    "queries": [
                        {
                            "sql": query.text,
                            "frequency": int(query.frequency),
                            "tables_columns": self._extract_tables_columns(query)
                        } for query in workload.queries
                    ],
                    "labels": [
                        {
                            "table": index.table().name,
                            "columns": [col.name for col in index.columns]
                        } for index in recommended_indexes
                    ],
                    "cost_improvement": data["cost_improvement"]
                }
                test_output.append(workload_data)

            # 保存test workload文件
            test_file = f"{self.experiment_folder_path}/test_workloads_with_labels_{timestamp}.json"
            with open(test_file, 'w', encoding='utf-8') as f:
                json.dump(test_output, f, indent=2, ensure_ascii=False)
            logging.info(f"Test workloads with labels saved to: {test_file}")

    def _extract_tables_columns(self, query):
        """从query中提取tables_columns信息"""
        tables_columns = {}
        for column in query.columns:
            table_name = column.table.name
            column_name = column.name
            if table_name not in tables_columns:
                tables_columns[table_name] = []
            if column_name not in tables_columns[table_name]:
                tables_columns[table_name].append(column_name)
        return tables_columns
