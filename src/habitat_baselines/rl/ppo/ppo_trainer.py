#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat_baselines.rl.ppo.ppo import PPONonOracle
import json
import os
import time
import pathlib
import sys
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from einops import rearrange
from matplotlib import pyplot as plt
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter
import tqdm
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config
from habitat.core.logging import logger
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainerNonOracle, BaseRLTrainerOracle
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.rollout_storage import RolloutStorageOracle, RolloutStorageNonOracle
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
)
from habitat_baselines.rl.ppo import PPONonOracle, PPOOracle, BaselinePolicyNonOracle, BaselinePolicyOracle, ProposedPolicyOracle
from utils.log_manager import LogManager
from utils.log_writer import LogWriter
from habitat.utils.visualizations import fog_of_war, maps


def to_grid(coordinate_min, coordinate_max, global_map_size, position):
    grid_size = (coordinate_max - coordinate_min) / global_map_size
    grid_x = ((coordinate_max - position[0]) / grid_size).round()
    grid_y = ((position[1] - coordinate_min) / grid_size).round()
    return int(grid_x), int(grid_y)


@baseline_registry.register_trainer(name="oracle")
class PPOTrainerO(BaseRLTrainerOracle):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        if config is not None:
            logger.info(f"config: {config}")

        self._static_encoder = False
        self._encoder = None
        
        self._num_picture = config.TASK_CONFIG.TASK.PICTURE.NUM_PICTURE
        #撮った写真のRGB画像を保存
        #self._taken_picture = []
        #撮った写真のciと位置情報、向きを保存
        self._taken_picture_list = []
        
        self._observed_object_ci = []
        self._target_index_list = []
        self._taken_index_list = []
        
        self.TARGET_THRESHOLD = 250
        self._dis_pre = []


    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        self.actor_critic = ProposedPolicyOracle(
            agent_type = self.config.TRAINER_NAME,
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            device=self.device,
            previous_action_embedding_size=self.config.RL.PREVIOUS_ACTION_EMBEDDING_SIZE,
            use_previous_action=self.config.RL.PREVIOUS_ACTION
        )
        self.actor_critic.to(self.device)

        self.agent = PPOOracle(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision", "traj_metrics", "ci"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            """
            if k == "ci":
                result[k] = float(v[0])
            """
                
            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results
    
    
    def _delete_observed_target(self, n):
        for i in self._target_index_list[n]:
            if self._observed_object_ci[n][i-maps.MAP_TARGET_POINT_INDICATOR] > self.TARGET_THRESHOLD:
                self._target_index_list[n].remove(i)

    
    def _do_take_picture_object(self, top_down_map, fog_of_war_map, n):
        # maps.MAP_TARGET_POINT_INDICATOR(6)が写真の中に何グリッドあるかを返す
        ci = 0
        for i in range(len(top_down_map[n])):
            for j in range(len(top_down_map[n][0])):
                if fog_of_war_map[n][i][j] == 1:
                    if top_down_map[n][i][j] in self._target_index_list[n]:
                        ci += 1
                        self._observed_object_ci[n][top_down_map[n][i][j]-maps.MAP_TARGET_POINT_INDICATOR]+=1
                        if top_down_map[n][i][j] not in self._taken_index_list[n]:
                            self._taken_index_list[n].append(top_down_map[n][i][j])

        # ciが閾値を超えているobjectがあれば削除
        self._delete_observed_target(n)
        
        # もし全部のobjectが削除されたら、リセット
        if len(self._target_index_list[n]) == 0:
            self._target_index_list[n] = [maps.MAP_TARGET_POINT_INDICATOR, maps.MAP_TARGET_POINT_INDICATOR+1, maps.MAP_TARGET_POINT_INDICATOR+2]
            self._observed_object_ci[n] = [0, 0, 0]
            
        return ci
            
            
    # 写真を撮った範囲のマップを作成
    def _create_picture_range_map(self, top_down_map, fog_of_war_map):
        # 0: 壁など, 1: 写真を撮った範囲, 2: 巡回可能領域
        picture_range_map = np.zeros_like(top_down_map)
        for i in range(len(top_down_map)):
            for j in range(len(top_down_map[0])):
                if top_down_map[i][j] != 0:
                    if fog_of_war_map[i][j] == 1:
                        picture_range_map[i][j] = 1
                    else:
                        picture_range_map[i][j] = 2
                        
        return picture_range_map
            
    # fog_mapがpre_fog_mapと閾値以上の割合で被っているか
    def _check_percentage_of_fog(self, fog_map, pre_fog_map, threshold=0.25):
        y = len(fog_map)
        x = len(fog_map[0])
        
        num = 0 #fog_mapのMAP_VALID_POINTの数
        num_covered = 0 #pre_fog_mapと被っているグリッド数
        
        y_pre = len(pre_fog_map)
        x_pre = len(pre_fog_map[0])
        
        
        if (x==x_pre) and (y==y_pre):
            for i in range(y):
                for j in range(x):
                    # fog_mapで写真を撮っている範囲の時
                    if fog_map[i][j] == 1:
                        num += 1
                        # fogとpre_fogがかぶっている時
                        if pre_fog_map[i][j] == 1:
                            num_covered += 1
                            
            if num == 0:
                per = 0.0
            else:
                per = num_covered / num
            
            if per < threshold:
                return False
            else:
                return True
        else:
            False
        
    # fog_mapがidx以外のpre_fog_mapと被っている割合を算出
    def _cal_rate_of_fog_other(self, fog_map, pre_fog_of_war_map_list, cover_list, idx):
        y = len(fog_map)
        x = len(fog_map[0])
        
        num = 0.0 #fog_mapのMAP_VALID_POINTの数
        num_covered = 0.0 #pre_fog_mapのどれかと被っているグリッド数
        
        for i in range(y):
            for j in range(x):
                # fog_mapで写真を撮っている範囲の時
                if fog_map[i][j] == 1:
                    num += 1
                    
                    # 被っているmapを検査する
                    for k in range(len(cover_list)):
                        map_idx = cover_list[k]
                        if map_idx == idx:
                            continue
                        
                        pre_map = pre_fog_of_war_map_list[map_idx]
                        # fogとpre_fogがかぶっている時
                        if pre_map[i][j] == 1:
                            num_covered += 1
                            break
                        
        if num == 0:
            rate = 0.0
        else:
            rate = num_covered / num
        
        return rate
    
    
    def _compare_with_changed_CI(self, picture_range_map, pre_fog_of_war_map_list, cover_list, ci, pre_ci, idx):
        rate = self._cal_rate_of_fog_other(picture_range_map, pre_fog_of_war_map_list, cover_list, idx)
        ci = ci * (1-rate) # k以外と被っている割合分小さくする
        if ci > pre_ci:
            return True
        else:
            return False
        

    def _exec_kachaka(self, log_manager, date) -> None:
        #ログ出力設定
        self.log_manager = log_manager
        #time, reward
        eval_reward_logger = self.log_manager.createLogWriter("reward")
        #time, ci, exp_area, distance. path_length
        eval_metrics_logger = self.log_manager.createLogWriter("metrics")
        
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        

        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"
            
            
        # evaluate multiple checkpoints in order
        checkpoint_index = 81
        while True:
            checkpoint_path = None
            while checkpoint_path is None:
                checkpoint_path = poll_checkpoint_folder(
                    self.config.EVAL_CKPT_PATH_DIR, checkpoint_index
                )
            logger.info(f"=======current_ckpt: {checkpoint_path}=======")
            checkpoint_index += 1
        
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        print("PATH")
        print(checkpoint_path)

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        self._setup_actor_critic_agent(ppo_cfg)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic
        
        self._taken_picture = []
        self._taken_picture_list = []
        self._target_index_list = []
        self._taken_index_list = []
        self._observed_object_ci = []
        for i in range(self.envs.num_envs):
            self._taken_picture.append([])
            self._taken_picture_list.append([])
            self._target_index_list.append([maps.MAP_TARGET_POINT_INDICATOR, maps.MAP_TARGET_POINT_INDICATOR+1, maps.MAP_TARGET_POINT_INDICATOR+2])
            self._taken_index_list.append([])
            self._observed_object_ci.append([0, 0, 0])
            self._dis_pre.append(-1)
        
        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )
        current_episode_exp_area = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )
        current_episode_distance = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )
        current_episode_ci = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )
        
        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes = dict()  # dict of dicts that stores stats per episode
        raw_metrics_episodes = dict()

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR+"/"+date, exist_ok=True)

        pbar = tqdm.tqdm(total=self.config.TEST_EPISODE_COUNT)
        self.actor_critic.eval()
        while (
            len(stats_episodes) < self.config.TEST_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                prev_actions.copy_(actions)

            outputs = self.envs.step([a[0].item() for a in actions])
 
            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, device=self.device)
            
            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )
            
            reward = []
            ci = []
            exp_area = [] # 探索済みのエリア()
            exp_area_pre = []
            distance = []
            fog_of_war_map = []
            top_down_map = [] 
            top_map = []
            n_envs = self.envs.num_envs
            
            for i in range(n_envs):
                reward.append(rewards[i][0][0])
                ci.append(rewards[i][0][1])
                exp_area.append(rewards[i][0][2]-rewards[i][0][3])
                exp_area_pre.append(rewards[i][0][3])
                fog_of_war_map.append(infos[i]["picture_range_map"]["fog_of_war_mask"])
                top_down_map.append(infos[i]["picture_range_map"]["map"])
                top_map.append(infos[i]["top_down_map"]["map"])
                
                # multi goal distanceの計算
                dis = 0.0
                if self._dis_pre[i] == -1:
                    self._dis_pre[i] = 0
                    for j in range(3):
                        self._dis_pre[i] += rewards[i][0][5][j]
                for j in self._target_index_list[i]:
                    dis += rewards[i][0][4][j-maps.MAP_TARGET_POINT_INDICATOR]
            
                reward[i] += self._dis_pre[i] - dis
                distance.append(self._dis_pre[i] - dis)
                self._dis_pre[i] = dis
            
            for n in range(len(observations)):
            #TAKE_PICTUREが呼び出されたかを検証
                if ci[n] != -sys.float_info.max:
                    # 今回撮ったpicture(p_n)が保存してあるpicture(p_k)とかぶっているkを保存
                    cover_list = [] 
                    picture_range_map = self._create_picture_range_map(top_down_map[n], fog_of_war_map[n])
                    
                    ci[n] = self._do_take_picture_object(top_map, fog_of_war_map, n)

                    if ci[n] == 0:
                        continue
                    
                    # p_kのそれぞれのpicture_range_mapのリスト
                    pre_fog_of_war_map = [sublist[1] for sublist in self._taken_picture_list[n]]
                        
                    # それぞれと閾値より被っているか計算
                    idx = -1
                    min_ci = ci[n]
                    for k in range(len(pre_fog_of_war_map)):
                        # 閾値よりも被っていたらcover_listにkを追加
                        if self._check_percentage_of_fog(picture_range_map, pre_fog_of_war_map[k]) == True:
                            cover_list.append(k)
                                
                        #ciの最小値の写真を探索(１つも被っていない時用)
                        if min_ci < self._taken_picture_list[n][idx][0]:
                            idx = k
                            min_ci = self._taken_picture_list[n][idx][0]
                            
                    # 今までの写真と多くは被っていない時
                    if len(cover_list) == 0:
                        #範囲が多く被っていなくて、self._num_picture回未満写真を撮っていたらそのまま保存
                        if len(self._taken_picture_list[n]) != self._num_picture:
                            self._taken_picture_list[n].append([ci[n], picture_range_map])
                            if len(self.config.VIDEO_OPTION) > 0:
                                self._taken_picture[n].append(observations[n]["rgb"])
                            reward[n] += ci[n]
                                
                        #範囲が多く被っていなくて、self._num_picture回以上写真を撮っていたら
                        else:
                            # 今回の写真が保存してある写真の１つでもCIが高かったらCIが最小の保存写真と入れ替え
                            if idx != -1:
                                ci_pre = self._taken_picture_list[n][idx][0]
                                self._taken_picture_list[n][idx] = [ci[n], picture_range_map]
                                if len(self.config.VIDEO_OPTION) > 0:
                                    self._taken_picture[n][idx] = observations[n]["rgb"]   
                                reward[n] += (ci[n] - ci_pre) 
                                ci[n] -= ci_pre
                            else:
                                ci[n] = 0.0    
                            
                    # 1つとでも多く被っていた時    
                    else:
                        min_idx = -1
                        min_ci_k = 1000
                        # 多く被った写真のうち、ciが最小のものを計算
                        for k in range(len(cover_list)):
                            idx_k = cover_list[k]
                            if self._taken_picture_list[n][idx_k][0] < min_ci_k:
                                min_ci_k = self._taken_picture_list[n][idx_k][0]
                                min_idx = idx_k
                                    
                        # 被った割合分小さくなったCIでも保存写真の中の最小のCIより大きかったら交換
                        if self._compare_with_changed_CI(picture_range_map, pre_fog_of_war_map, cover_list, ci[n], min_ci_k, min_idx) == True:
                            self._taken_picture_list[n][min_idx] = [ci[n], picture_range_map]
                            if len(self.config.VIDEO_OPTION) > 0:
                                self._taken_picture[n][min_idx] = observations[n]["rgb"]   
                            reward[n] += (ci[n] - min_ci_k)  
                            ci[n] -= min_ci_k
                        else:
                            ci[n] = 0.0
                else:
                    ci[n] = 0.0
                
            reward = torch.tensor(
                reward, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            #exp_area = np.array(exp_area)
            exp_area = torch.tensor(
                exp_area, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            distance = torch.tensor(
                distance, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            #ci = np.array(ci)
            ci= torch.tensor(
                ci, dtype=torch.float, device=self.device
            ).unsqueeze(1)

            current_episode_reward += reward
            current_episode_exp_area += exp_area
            current_episode_distance += distance
            current_episode_ci += ci
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []

            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:                    
                    pbar.update()
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats["exp_area"] = current_episode_exp_area[i].item()
                    episode_stats["distance"] = current_episode_distance[i].item()
                    episode_stats["ci"] = current_episode_ci[i].item()
                    
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    current_episode_exp_area[i] = 0
                    current_episode_distance[i] = 0
                    current_episode_ci[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats
                    
                    raw_metrics_episodes[
                        current_episodes[i].scene_id + '.' + 
                        current_episodes[i].episode_id
                    ] = infos[i]["raw_metrics"]

                    if len(self.config.VIDEO_OPTION) > 0:
                        if len(rgb_frames[i]) == 0:
                            frame = observations_to_image(observations[i], infos[i], actions[i].cpu().numpy())
                            rgb_frames[i].append(frame)
                        picture = rgb_frames[i][-1]
                        for j in range(50):
                           rgb_frames[i].append(picture) 
                        metrics=self._extract_scalars_from_info(infos[i])
                        name_ci = 0.0
                        
                        for j in range(len(self._taken_picture_list[i])):
                            name_ci += self._taken_picture_list[i][j][0]
                        
                        name_ci = str(name_ci) + "-" + str(len(stats_episodes))
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR+"/"+date,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=metrics,
                            tb_writer=writer,
                            name_ci=name_ci,
                        )
        
                        # Save taken picture
                        metric_strs = []
                        in_metric = ['exp_area', 'ci', 'distance']
                        for k, v in metrics.items():
                            if k in in_metric:
                                metric_strs.append(f"{k}={v:.2f}")
                        
                        name_p = 0.0  
                            
                        for j in range(len(self._taken_picture_list[i])):
                            eval_picture_top_logger = self.log_manager.createLogWriter("picture_top_" + str(current_episodes[i].episode_id) + "_" + str(j) + "_" + str(checkpoint_index))
                
                            for k in range(len(self._taken_picture_list[i][j][1])):
                                for l in range(len(self._taken_picture_list[i][j][1][0])):
                                    eval_picture_top_logger.write(str(self._taken_picture_list[i][j][1][k][l]))
                                eval_picture_top_logger.writeLine()
                                
                            name_p = self._taken_picture_list[i][j][0]
                            picture_name = "episode=" + str(current_episodes[i].episode_id)+ "-ckpt=" + str(checkpoint_index) + "-" + str(j) + "-" + str(name_p)
                            dir_name = "./taken_picture/" + date 
                            if not os.path.exists(dir_name):
                                os.makedirs(dir_name)
                        
                            picture = self._taken_picture[i][j]
                            plt.figure()
                            ax = plt.subplot(1, 1, 1)
                            ax.axis("off")
                            plt.imshow(picture)
                            plt.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.95)
                            path = dir_name + "/" + picture_name + ".png"
                        
                            plt.savefig(path)
                            logger.info("Picture created: " + path)
                            
                        rgb_frames[i] = []
                        
                    self._taken_picture[i] = []
                    self._taken_picture_list[i] = []
                    self._target_index_list[i] = [maps.MAP_TARGET_POINT_INDICATOR, maps.MAP_TARGET_POINT_INDICATOR+1, maps.MAP_TARGET_POINT_INDICATOR+2]
                    self._taken_index_list[i] = []
                    self._observed_object_ci[i] = [0, 0, 0]
                    self._dis_pre[i] = -1

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i], actions[i].cpu().numpy())
                    rgb_frames[i].append(frame)

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                current_episode_exp_area,
                current_episode_distance,
                current_episode_ci,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                current_episode_exp_area,
                current_episode_distance,
                current_episode_ci,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()])
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")
        


        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]
        
        eval_reward_logger.writeLine(str(step_id) + "," + str(aggregated_stats["reward"]))

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}

        logger.info("CI:" + str(metrics["ci"]))
        eval_metrics_logger.writeLine(str(step_id) + "," + str(metrics["ci"]) + "," + str(metrics["exp_area"]) + "," + str(metrics["distance"]) + "," + str(metrics["raw_metrics.agent_path_length"]))

        self.envs.close()
