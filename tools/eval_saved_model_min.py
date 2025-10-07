import argparse
import json
import os
import importlib
import time
import sys
import logging
from datetime import datetime

# Ensure repository root is on sys.path so local imports (swirl, gym_db, etc.) work
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from swirl.experiment import Experiment
from gym_db.common import EnvironmentType


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved SWIRL model against a workload")
    parser.add_argument("--model-dir", dest="model_dir", default="/home/baiyutao/SWIRL/experiment_results/ID_MixTrain_fixVec",
                        help="Path to the experiment folder containing saved model files (default: %(default)s)")
    parser.add_argument("--config", dest="config_path", default="/home/baiyutao/SWIRL/experiments/localtest.json",
                        help="Path to an experiment configuration JSON (default: %(default)s)")
    parser.add_argument("--workload", dest="workload_path", default="/home/baiyutao/SWIRL/LocalExp/dsb.7000.100w.10q.workload.json",
                        help="Path to workload JSON to evaluate (overrides config) (default: %(default)s)")
    parser.add_argument("--timesteps", dest="timesteps", type=int, default=1,
                        help="Number of timesteps to run for quick evaluations (default: %(default)s)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    model_dir = args.model_dir
    config_path = args.config_path
    dsb_path = args.workload_path

    logging.info(f"Using config: {config_path}")
    logging.info(f"Using workload: {dsb_path}")
    logging.info(f"Model directory: {model_dir}")

    # 读取原配置并写入临时配置文件，分配唯一实验ID，避免覆盖已有目录
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    # 调整配置以便快速评估
    cfg["ExternalWorkload"] = True
    cfg["WorkloadPath"] = dsb_path
    cfg["TestExternalWorkload"] = True
    cfg["TestWorkloadPath"] = dsb_path
    cfg["workload"]["size"] = 1
    cfg["workload"]["training_instances"] = 10
    cfg["workload"]["validation_testing"]["number_of_workloads"] = 10
    cfg["timesteps"] = args.timesteps

    unique_id = f"{cfg.get('id', 'Eval')}_DSB_Eval_{int(time.time())}"
    cfg["id"] = unique_id

    tmp_cfg_path = os.path.join(os.path.dirname(config_path), f".__tmp_{unique_id}.json")
    with open(tmp_cfg_path, 'w') as f:
        json.dump(cfg, f)

    logging.info(f"Created temporary config: {tmp_cfg_path}")

    exp = Experiment(tmp_cfg_path)
    exp.prepare()

    algo_mod = (
        "stable_baselines"
        if exp.config["rl_algorithm"]["stable_baselines_version"] == 2
        else "stable_baselines3"
    )
    Algo = getattr(importlib.import_module(algo_mod), exp.config["rl_algorithm"]["algorithm"])

    # 查找模型文件（优先 best_mean_reward_model、其次 final_model）
    model_path = os.path.join(model_dir, "best_mean_reward_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "final_model.zip")
    if not os.path.exists(model_path):
        logging.error(f"未找到模型文件 in {model_dir}. Expected best_mean_reward_model.zip or final_model.zip")
        sys.exit(2)

    logging.info(f"Loading model from: {model_path}")
    model = Algo.load(model_path)

    # 为模型构建并设置测试环境；尝试恢复训练时保存的 VecNormalize（如果存在），
    # 以便使用相同的 observation/reward 归一化统计量；如果不存在则创建新的。
    base_env = exp.DummyVecEnv([exp.make_env(0, EnvironmentType.TESTING)])
    vec_loaded = False
    vec_env = exp.VecNormalize(
        base_env,
        norm_obs=True,
        norm_reward=False,
        gamma=exp.config["rl_algorithm"]["gamma"],
        training=False,
    )
    logging.info("Created new VecNormalize for evaluation")

    model.set_env(vec_env)
    exp.set_model(model)

    logging.info("Starting evaluation...")
    start_ts = datetime.now()
    results, _ = exp.test_model(model)
    end_ts = datetime.now()

    perfs = [r[1] for r in results]

    output = {
        "timestamp": start_ts.isoformat(),
        "model": model_path,
        "vec_normalize_loaded": vec_loaded,
        "workload_path": dsb_path,
        "num_workloads": len(perfs),
        "mean_performances": perfs,
        "overall_mean": (sum(perfs) / len(perfs)) if perfs else None,
        "duration_seconds": (end_ts - start_ts).total_seconds(),
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))
    # 打印更可读的摘要
    logging.info(f"Evaluated {output['num_workloads']} workloads in {output['duration_seconds']:.2f}s")
    logging.info(f"Overall mean performance: {output['overall_mean']}")


if __name__ == "__main__":
    main()


