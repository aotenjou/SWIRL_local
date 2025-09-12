import collections
import copy
import logging
import os
import random
import time

import gym

from gym_db.common import EnvironmentType
from index_selection_evaluation.selection.cost_evaluation import CostEvaluation
from index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector
from index_selection_evaluation.selection.index import Index
from index_selection_evaluation.selection.utils import b_to_mb


class DBEnvV1(gym.Env):
    def __init__(self, environment_type=EnvironmentType.TRAINING, config=None):
        super(DBEnvV1, self).__init__()

        self.rnd = random.Random()
        self.rnd.seed(config["random_seed"])
        self.env_id = config["env_id"]
        self.environment_type = environment_type
        self.config = config

        self.number_of_resets = 0
        self.total_number_of_steps = 0

        # Get database connection parameters from environment variables
        db_host = os.getenv('DATABASE_HOST', 'localhost')
        db_port = os.getenv('DATABASE_PORT', '54321')
        self.connector = PostgresDatabaseConnector(config["database_name"], autocommit=True, host=db_host, port=db_port)
        self.connector.drop_indexes()
        self.cost_evaluation = CostEvaluation(self.connector)

        self.globally_indexable_columns = config["globally_indexable_columns"]
        # In certain cases, workloads are consumed: therefore, we need copy
        self.workloads = copy.copy(config["workloads"])
        self.current_workload_idx = 0
        self.similar_workloads = config["similar_workloads"]
        self.max_steps_per_episode = config["max_steps_per_episode"]

        self.action_manager = config["action_manager"]
        self.action_manager.test_variable = self.env_id
        self.action_space = self.action_manager.get_action_space()

        self.observation_manager = config["observation_manager"]
        self.observation_space = self.observation_manager.get_observation_space()

        self.reward_calculator = config["reward_calculator"]

        self._init_modifiable_state()

        if self.environment_type != environment_type.TRAINING:
            self.episode_performances = collections.deque(maxlen=len(config["workloads"]))
            # 添加索引选择顺序记录
            self.selected_indexes_sequence = []
            # 添加时间记录
            self.episode_start_time = None
            self.episode_end_time = None
            self.total_index_selection_time = 0.0

    def reset(self):
        self.number_of_resets += 1
        self.total_number_of_steps += self.steps_taken

        # 记录episode开始时间
        if self.environment_type != EnvironmentType.TRAINING:
            self.episode_start_time = time.time()

        initial_observation = self._init_modifiable_state()

        return initial_observation

    def _step_asserts(self, action):
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"
        assert (
            self.valid_actions[action] == self.action_manager.ALLOWED_ACTION
        ), f"Agent has chosen invalid action: {action}"
        assert (
            Index(self.globally_indexable_columns[action]) not in self.current_indexes
        ), f"{Index(self.globally_indexable_columns[action])} already in self.current_indexes"

    def step(self, action):
        self._step_asserts(action)

        self.steps_taken += 1
        old_index_size = 0

        new_index = Index(self.globally_indexable_columns[action])
        self.current_indexes.add(new_index)

        # 记录索引选择顺序
        if self.environment_type != EnvironmentType.TRAINING:
            # 确保索引大小不为None
            index_size = new_index.estimated_size if new_index.estimated_size is not None else 0
            old_size = old_index_size if old_index_size is not None else 0
            
            self.selected_indexes_sequence.append({
                'step': self.steps_taken,
                'action': action,
                'index': new_index,
                'index_size': index_size,
                'current_storage': self.current_storage_consumption + index_size - old_size
            })

        if not new_index.is_single_column():
            parent_index = Index(new_index.columns[:-1])

            for index in self.current_indexes:
                if index == parent_index:
                    old_index_size = index.estimated_size if index.estimated_size is not None else 0

            self.current_indexes.remove(parent_index)

            assert old_index_size >= 0, "Parent index size must have been found if not single column index."

        environment_state = self._update_return_env_state(
            init=False, new_index=new_index, old_index_size=old_index_size
        )
        current_observation = self.observation_manager.get_observation(environment_state)

        self.valid_actions, is_valid_action_left = self.action_manager.update_valid_actions(
            action, self.current_budget, self.current_storage_consumption
        )
        episode_done = self.steps_taken >= self.max_steps_per_episode or not is_valid_action_left

        reward = self.reward_calculator.calculate_reward(environment_state)

        if episode_done and self.environment_type != EnvironmentType.TRAINING:
            # 记录episode结束时间
            self.episode_end_time = time.time()
            # 安全地计算时间差，确保start_time不为None
            if self.episode_start_time is not None:
                self.total_index_selection_time = self.episode_end_time - self.episode_start_time
            else:
                self.total_index_selection_time = 0.0
                logging.warning(f"episode_start_time was None for workload {self.current_workload}, using default time")
            
            self._report_episode_performance(environment_state)
            self.current_workload_idx += 1
            # print(f"Indexes: {len(self.current_indexes)}")

        return current_observation, reward, episode_done, {"action_mask": self.valid_actions}

    def _report_episode_performance(self, environment_state):
        episode_performance = {
            "achieved_cost": self.current_costs / self.initial_costs * 100,
            "memory_consumption": self.current_storage_consumption,
            "available_budget": self.current_budget,
            "evaluated_workload": self.current_workload,
            "indexes": self.current_indexes,
            "selected_indexes_sequence": self.selected_indexes_sequence.copy(),  # 添加索引选择顺序
            "index_selection_time": self.total_index_selection_time,  # 添加索引选择时间
        }

        # 增强输出信息，包含索引选择顺序和时间
        index_sequence_str = ""
        if self.selected_indexes_sequence:
            index_sequence_str = "\n    Index selection sequence:\n"
            for i, seq_item in enumerate(self.selected_indexes_sequence):
                index_sequence_str += f"    Step {seq_item['step']}: {seq_item['index']} (size: {b_to_mb(seq_item['index_size']):.2f}MB)\n"

        output = (
            f"Evaluated Workload ({self.environment_type}): {self.current_workload}\n    "
            f"Initial cost: {self.initial_costs:,.2f}, now: {self.current_costs:,.2f} "
            f"({episode_performance['achieved_cost']:.2f}). Reward: {self.reward_calculator.accumulated_reward}.\n    "
            f"Size: {b_to_mb(self.current_storage_consumption):.2f} with {len(self.current_indexes)} indexes:\n    "
            f"{self.current_indexes}\n    "
            f"Index selection time: {self.total_index_selection_time:.4f} seconds\n    "
            f"{index_sequence_str}"
        )
        logging.warning(output)

        self.episode_performances.append(episode_performance)

    def _init_modifiable_state(self):
        self.current_indexes = set()
        self.steps_taken = 0
        self.current_storage_consumption = 0
        self.reward_calculator.reset()
        
        # 重置索引选择顺序和时间
        if self.environment_type != EnvironmentType.TRAINING:
            self.selected_indexes_sequence = []
            self.episode_start_time = None
            self.episode_end_time = None
            self.total_index_selection_time = 0.0

        if len(self.workloads) == 0:
            self.workloads = copy.copy(self.config["workloads"])

        if self.environment_type == EnvironmentType.TRAINING:
            if self.similar_workloads:
                # Calculate step size to distribute workloads across parallel environments
                # Ensure we don't exceed the available workloads
                step_size = max(1, len(self.workloads) // 16)  # Assume max 16 parallel envs
                workload_idx = (self.env_id * step_size) % len(self.workloads)
                self.current_workload = self.workloads.pop(workload_idx)
            else:
                self.current_workload = self.rnd.choice(self.workloads)
        else:
            self.current_workload = self.workloads[self.current_workload_idx % len(self.workloads)]

        self.current_budget = self.current_workload.budget
        self.previous_cost = None

        self.valid_actions = self.action_manager.get_initial_valid_actions(self.current_workload, self.current_budget)
        environment_state = self._update_return_env_state(init=True)

        state_fix_for_episode = {
            "budget": self.current_budget,
            "workload": self.current_workload,
            "initial_cost": self.initial_costs,
        }
        self.observation_manager.init_episode(state_fix_for_episode)

        initial_observation = self.observation_manager.get_observation(environment_state)

        return initial_observation

    def _update_return_env_state(self, init, new_index=None, old_index_size=None):
        total_costs, plans_per_query, costs_per_query = self.cost_evaluation.calculate_cost_and_plans(
            self.current_workload, self.current_indexes, store_size=True
        )

        if not init:
            self.previous_cost = self.current_costs
            self.previous_storage_consumption = self.current_storage_consumption

        self.current_costs = total_costs

        if init:
            self.initial_costs = total_costs

        new_index_size = None

        if new_index is not None:
            self.current_storage_consumption += new_index.estimated_size if new_index.estimated_size is not None else 0
            self.current_storage_consumption -= old_index_size

            # This assumes that old_index_size is not None if new_index is not None
            new_size = new_index.estimated_size if new_index.estimated_size is not None else 0
            assert new_size >= old_index_size

            new_index_size = (new_index.estimated_size if new_index.estimated_size is not None else 0) - old_index_size
            if new_index_size == 0:
                new_index_size = 1

            if self.current_budget:
                assert b_to_mb(self.current_storage_consumption) <= self.current_budget, (
                    "Storage consumption exceeds budget: "
                    f"{b_to_mb(self.current_storage_consumption)} "
                    f" > {self.current_budget}"
                )

        environment_state = {
            "action_status": self.action_manager.current_action_status,
            "current_storage_consumption": self.current_storage_consumption,
            "current_cost": self.current_costs,
            "previous_cost": self.previous_cost,
            "initial_cost": self.initial_costs,
            "new_index_size": new_index_size,
            "plans_per_query": plans_per_query,
            "costs_per_query": costs_per_query,
        }

        return environment_state

    def get_cost_eval_cache_info(self):
        return (
            self.cost_evaluation.cost_requests,
            self.cost_evaluation.cache_hits,
            self.cost_evaluation.costing_time,
        )

    def get_cost_eval_cache(self):
        return self.cost_evaluation.cache

    def render(self, mode="human"):
        pass

    def close(self):
        pass
