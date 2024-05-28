#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import cv2
import sys
from PIL import Image
from collections import defaultdict
from typing import Any, Dict, List
import pyrealsense2 as rs
import clip
from sentence_transformers import SentenceTransformer, util
from lavis.models import load_model_and_preprocess

import numpy as np
import torch
import tqdm

from habitat import Config
from habitat.core.logging import logger
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainerOracle
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_utils import construct_env
from habitat_baselines.common.rollout_storage import RolloutStorageOracle
from habitat_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
    poll_checkpoint_folder
)
from habitat_baselines.rl.ppo import PPOOracle, ProposedPolicyOracle

import kachaka_api
sys.path.append(f"/home/{os.environ['USER']}/Desktop/habitat2kachaka/kachaka-api/python/")



def to_grid(client, realworld_x, realworld_y):
    map = client.get_png_map()
    grid_resolution = map.resolution
        
    # マップの原点からエージェントまでの距離を算出
    dx = realworld_x - map.origin.x
    dy = realworld_y - map.origin.y
        
    # エージェントのグリッド座標を求める
    grid_x = dx / map.resolution
    grid_y = dy / map.resolution
    grid_y = map.height-grid_y
        
    # resizeの分、割る
    grid_x /= 3
    grid_y /= 3
        
    # 四捨五入する
    grid_x = int(grid_x)
    grid_y = int(grid_y)
        
    return grid_x, grid_y


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
        self.env = None
        if config is not None:
            logger.info(f"config: {config}")

        self._static_encoder = False
        self._encoder = None
        
        self._num_picture = config.TASK_CONFIG.TASK.PICTURE.NUM_PICTURE
        #撮った写真のRGB画像を保存
        self._taken_picture = []
        #撮った写真のsaliencyとrange_mapを保存
        self._taken_picture_list = []
        
        self._observed_object_ci = []
        self._target_index_list = []
        self._taken_index_list = []
        
        self.TARGET_THRESHOLD = 250
        self._dis_pre = []
        
        # ストリームの設定
        realsense_config = rs.config()
        realsense_config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 15)
        realsense_config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 15)

        # ストリーミング開始
        self.pipeline = rs.pipeline()
        #self.pipeline.stop()
        self.pipeline.start(realsense_config)
        
        align_to = rs.stream.color
        self.align = rs.align(align_to)


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
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
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

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision", "traj_metrics", "saliency"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue
                
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
    
    def _create_caption(self, picture):
        # 画像からcaptionを生成する
        image = Image.fromarray(picture)
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        generated_text = self.lavis_model.generate({"image": image}, use_nucleus_sampling=True,num_captions=1)[0]
        return generated_text
    
    def create_description(self, picture_list):
        # captionを連結してdescriptionを生成する
        description = ""
        
        for i in range(len(picture_list)):
            description += (picture_list[i][3] + ". ")
            
        return description
    
    def _create_new_description_embedding(self, caption):
        # captionのembeddingを作成
        embedding = self.bert_model.encode(caption, convert_to_tensor=True)
        return embedding
    
    def _create_new_image_embedding(self, obs):
        image = Image.fromarray(obs)
        image = self.preprocess(image)
        image = torch.tensor(image).to(self.device).unsqueeze(0)
        embetting = self.clip_model.encode_image(image).float()
        return embetting
    
    def _cal_remove_index(self, picture_list, new_emmbedding):
        # 削除する写真を決める
        # 他のsyasinnとの類似度を計算し、合計が最大のものを削除
        
        sim_list = [[-10 for _ in range(len(picture_list)+1)] for _ in range(len(picture_list)+1)]
        sim_list[len(picture_list)][len(picture_list)] = 0.0
        for i in range(len(picture_list)):
            emd = picture_list[i][2]
            sim_list[i][len(picture_list)] = util.pytorch_cos_sim(emd, new_emmbedding).item()
            sim_list[len(picture_list)][i] = sim_list[i][len(picture_list)]
            for j in range(i, len(picture_list)):
                if i == j:
                    sim_list[i][j] = 0.0
                    continue
                    
                #logger.info(f"len: {len(picture_list)}, i: {i}, j: {j}")
                sim_list[i][j] = util.pytorch_cos_sim(emd, picture_list[j][2]).item()
                sim_list[j][i] = sim_list[i][j]
                
        total_sim = [sum(similarity_list) for similarity_list in sim_list]
        remove_index = total_sim.index(max(total_sim))
        return remove_index

    def _calculate_pic_sim(self, picture_list):
        if len(picture_list) <= 1:
            return 0.0
            
        sim_list = [[-10 for _ in range(len(picture_list))] for _ in range(len(picture_list))]

        for i in range(len(picture_list)):
            emd = picture_list[i][2]
            for j in range(i, len(picture_list)):
                if i == j:
                    sim_list[i][j] = 0.0
                    continue
                    
                sim_list[i][j] = util.pytorch_cos_sim(emd, picture_list[j][2]).item()
                sim_list[j][i] = sim_list[i][j]
                
        total_sim = np.sum(sim_list)
        total_sim /= (len(picture_list)*(len(picture_list)-1))
        return total_sim
    
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
        
        per = -1.0
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
                return False, per
            else:
                return True, per
        else:
            logger.info("CHECK, false")
            return False, per
        
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
    
    def _compareWithChangedSal(self, picture_range_map, pre_fog_of_war_map_list, cover_list, saliency, pre_saliency, idx):
        rate = self._cal_rate_of_fog_other(picture_range_map, pre_fog_of_war_map_list, cover_list, idx)
        saliency = saliency * (1-rate) # k以外と被っている割合分小さくする
        #logger.info(f"LIST_SIZE: {len(cover_list)}, RATE: {rate}, saliency: {saliency}, pre_saliency: {pre_saliency}")
        if saliency > pre_saliency:
            return 0
        elif saliency == pre_saliency:
            return 1
        else:
            return 2

    #def _exec_kachaka(self, log_manager, date, ip) -> None:
    def _exec_kachaka(self, date, ip) -> None:
        max_step = 500
        #self.cap = cv2.VideoCapture(0)
        #_, frame = self.cap.read()
        #cv2.imshow('webカメラ', frame)
        #cv2.imwrite('photo.jpg', frame)

        for _ in range(20):
            # フレーム待ち
            frames = self.pipeline.wait_for_frames()
        
        frames = self.align.process(frames)    
        # RGB
        color_frame = frames.get_color_frame()
        # 深度
        depth_frame = frames.get_depth_frame()
            
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        color_image = Image.fromarray(color_image)
        
        depth_image = np.asanyarray(depth_frame.get_data())   
        # 2次元データをカラーマップに変換
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap = Image.fromarray(depth_colormap)
            
        color_image.save("color.png")
        depth_colormap.save("depth.png")
        print("create")
        # ストリーミング停止
        self.pipeline.stop()
        
        client = kachaka_api.KachakaApiClient(ip)
        client.update_resolver()
        # カチャカにshelfをstartに連れていく
        print("Get the shelf and Go to the Start")
        #sclient.move_shelf("S01", "L01")
        client.move_shelf("S01", "start")
        client.set_auto_homing_enabled(False)
        
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
        checkpoint_index = 108
        print("checkpoint_index=" + str(checkpoint_index))
        while True:
            checkpoint_path = None
            while checkpoint_path is None:
                checkpoint_path = poll_checkpoint_folder(
                    self.config.EVAL_CKPT_PATH_DIR, checkpoint_index
                )
                print("checkpoint_path=" + str(checkpoint_path))
            print("checkpoint_path=" + str(checkpoint_path))
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
            self.env = construct_env(config, client)
            self._setup_actor_critic_agent(ppo_cfg)

            self.agent.load_state_dict(ckpt_dict["state_dict"])
            self.actor_critic = self.agent.actor_critic
        
            self._taken_picture = []
            self._taken_picture_list = []
        
            self._taken_picture.append([])
            self._taken_picture_list.append([])
            
            # Sentence-BERTモデルの読み込み
            self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # lavisモデルの読み込み
            self.lavis_model, self.vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=self.device)
            self.bert_model.to(self.device)
            self.lavis_model.to(self.device)

            # Load the clip model
            self.clip_model, self.preprocess = clip.load('ViT-B/32', self.device)
        
            observations = self.env.reset()
            batch = batch_obs(observations, device=self.device)

            current_episode_reward = torch.zeros(1, 1, device=self.device)
            current_episode_exp_area = torch.zeros(1, 1, device=self.device)
            current_episode_picsim = torch.zeros(1, 1, device=self.device)
            current_episode_sum_saliency = torch.zeros(1, 1, device=self.device)
            
            
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

            rgb_frames = [[]]  # type: List[List[np.ndarray]]
            if len(self.config.VIDEO_OPTION) > 0:
                os.makedirs(self.config.VIDEO_DIR+"/"+date, exist_ok=True)

            pbar = tqdm.tqdm(total=self.config.TEST_EPISODE_COUNT)
            episode_stats = dict()
            self.actor_critic.eval()
            
            self.step = 0
            while (
                #len(stats_episodes) < self.config.TEST_EPISODE_COUNT
                self.step < max_step
            ):
                if self.step % 10 == 0:
                    print("---------------------")
                    client.speak(str(self.step) + "ステップ終了しました。")
                
                self.step+=1
                print(str(self.step) + ": ", end="")

                with torch.no_grad():
                    (
                        _,
                        action,
                        _,
                        test_recurrent_hidden_states,
                    ) = self.actor_critic.act(
                        batch,
                        test_recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                        deterministic=False,
                    )
                    
                    pre_ac = torch.zeros(prev_actions.shape[0], 1, device=self.device, dtype=torch.long)
                    for i in range(prev_actions.shape[0]):
                        pre_ac[i] = prev_actions[i]

                    prev_actions.copy_(action)

                outputs = self.env.step(action[0].item())
    
                observations, rewards, done, infos = outputs
                batch = batch_obs(observations, device=self.device)
                
                not_done_masks = torch.tensor(
                    [[0.0] if done else [1.0]],
                    dtype=torch.float,
                    device=self.device,
                )
                
                reward = []
                saliency = []
                pic_sim = []
                exp_area = [] # 探索済みのエリア()
                exp_area_pre = []
                fog_of_war_map = []
                top_down_map = [] 
                sum_saliency = []
                
                reward.append(rewards[0])
                saliency.append(rewards[1])
                pic_sim.append(0)
                exp_area.append(rewards[2]-rewards[3])
                exp_area_pre.append(rewards[3])
                fog_of_war_map.append(infos["picture_range_map"]["fog_of_war_mask"])
                top_down_map.append(infos["picture_range_map"]["map"])
                sum_saliency.append(0)
                    
                #TAKE_PICTUREが呼び出されたかを検証
                if saliency[0] == -1:
                    pass

                # 2回連続でTAKE_PICTUREをした場合は保存しない
                elif pre_ac[0].item() == action.item():
                    pass

                else:
                    # 今回撮ったpicture(p_n)が保存してあるpicture(p_k)とかぶっているkを保存
                    cover_list = [] 
                    cover_per_list = []
                    picture_range_map = self._create_picture_range_map(top_down_map[0], fog_of_war_map[0])
                    
                    picture_list = self._taken_picture_list[0]
                        
                    pred_description = self.create_description(picture_list)
                    
                    caption = self._create_caption(observations["rgb"])
                    #new_emmbedding = self._create_new_description_embedding(caption)
                    new_emmbedding = self._create_new_image_embedding(observations["rgb"])

                    # p_kのそれぞれのpicture_range_mapのリスト
                    pre_fog_of_war_map = [sublist[1] for sublist in picture_list]

                    # それぞれと閾値より被っているか計算
                    idx = -1
                    min_sal = saliency[0]

                    for k in range(len(pre_fog_of_war_map)):
                        # 閾値よりも被っていたらcover_listにkを追加
                        check, per = self._check_percentage_of_fog(picture_range_map, pre_fog_of_war_map[k], threshold=0.1)
                        cover_per_list.append(per)
                        if check == True:
                            cover_list.append(k)

                        #saliencyの最小値の写真を探索(１つも被っていない時用)
                        if (idx == -1) and (min_sal == picture_list[idx][0]):
                            idx = -2
                        elif min_sal > picture_list[idx][0]:
                            idx = k
                            min_sal = picture_list[idx][0]

                    # 今までの写真と多くは被っていない時
                    if len(cover_list) == 0:
                        #範囲が多く被っていなくて、self._num_picture回未満写真を撮っていたらそのまま保存
                        if len(picture_list) != self._num_picture:
                            picture_list.append([saliency[0], picture_range_map, new_emmbedding, caption])
                            self._taken_picture[0].append(observations["rgb"])
                            self._taken_picture_list[0] = picture_list
                            print("save picture: 0")

                        #範囲が多く被っていなくて、self._num_picture回以上写真を撮っていたら
                        else:
                            # 今回の写真が保存している写真でsaliencyが最小のものと同じだった場合、写真の類似度が最大のものと交換
                            if idx == -2:
                                remove_index = self._cal_remove_index(picture_list, new_emmbedding)
                                # 入れ替えしない場合
                                if remove_index == len(picture_list):
                                    print("not save picture: 0")
                                else:
                                    picture_list[remove_index] = [saliency[0], picture_range_map, new_emmbedding, caption]
                                    self._taken_picture_list[0] = picture_list
                                    self._taken_picture[0][remove_index] = observations["rgb"]
                                    print("change picture: 0")

                            # 今回の写真が保存してある写真の１つでもSaliencyが高かったらSaliencyが最小の保存写真と入れ替え
                            elif idx != -1:
                                picture_list[idx] = [saliency[0], picture_range_map, new_emmbedding, caption]
                                self._taken_picture_list[0] = picture_list
                                self._taken_picture[0][idx] = observations["rgb"]
                                print("change picture: 1")

                    # 1つとでも多く被った場合
                    else:
                        min_idx = -1
                        #min_sal_k = 1000
                        max_sal_k = 0.0
                        idx_sal = -1
                        # 多く被った写真のうち、saliencyが最小のものを計算
                        # 多く被った写真のうち、被っている割合が多い写真とsaliencyを比較
                        for k in range(len(cover_list)):
                            idx_k = cover_list[k]
                            if max_sal_k < cover_per_list[idx_k]:
                                max_sal_k = cover_per_list[idx_k]
                                min_idx = idx_k
                                idx_sal = picture_list[idx_k][0]

                        # 被った割合分小さくなったCIでも保存写真の中の最小のCIより大きかったら交換
                        #if self._compareWithChangedSal(picture_range_map, pre_fog_of_war_map, cover_list, saliency[0], min_sal_k, min_idx) == True:
                        res = self._compareWithChangedSal(picture_range_map, pre_fog_of_war_map, cover_list, saliency[0], idx_sal, min_idx)
                        if res == 0:
                            picture_list[min_idx] = [saliency[0], picture_range_map, new_emmbedding, caption]
                            self._taken_picture_list[0] = picture_list
                            self._taken_picture[0][min_idx] = observations["rgb"]  
                            print("change picture: 2") 
                        # 被った割合分小さくなったCIと保存写真の中の最小のCIが等しかったら写真の類似度が最大のものを削除
                        elif res == 1:
                            remove_index = self._cal_remove_index(picture_list, new_emmbedding)
                            # 入れ替えしない場合
                            if remove_index == len(picture_list):
                                print("not save picture: 1")
                            else:
                                picture_list[remove_index] = [saliency[0], picture_range_map, new_emmbedding, caption]
                                self._taken_picture_list[0] = picture_list
                                self._taken_picture[0][remove_index] = observations["rgb"]
                                print("change picture: 3")
                        else:
                            print("not save picture: 2")
                    
                reward = torch.tensor(reward, dtype=torch.float, device=self.device).unsqueeze(1)
                exp_area = torch.tensor(exp_area, dtype=torch.float, device=self.device).unsqueeze(1)
                pic_sim = torch.tensor(pic_sim, dtype=torch.float, device=current_episode_reward.device).unsqueeze(1)
                
                current_episode_reward += reward
                current_episode_exp_area += exp_area
                envs_to_pause = []
                
                # episode continues
                if len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations, infos, action.cpu().numpy())
                    for _ in range(10):
                        rgb_frames[0].append(frame)

                (
                    self.env,
                    test_recurrent_hidden_states,
                    not_done_masks,
                    current_episode_reward,
                    current_episode_exp_area,
                    current_episode_picsim,
                    current_episode_sum_saliency,
                    prev_actions,
                    batch,
                    rgb_frames,
                ) = self._pause_envs(
                    envs_to_pause,
                    self.env,
                    test_recurrent_hidden_states,
                    not_done_masks,
                    current_episode_reward,
                    current_episode_exp_area,
                    current_episode_picsim,
                    current_episode_sum_saliency,
                    prev_actions,
                    batch,
                    rgb_frames,
                )
                
                if self.step % 50 == 0:
                    pred_description = self.create_description(self._taken_picture_list[0])
                    pic_sim[0] = self._calculate_pic_sim(self._taken_picture_list[0])
                    current_episode_picsim[0] += pic_sim[0]

                    for j in range(len(self._taken_picture_list[0])):
                        sum_saliency[0] += self._taken_picture_list[0][j][0]
                    if len(self._taken_picture_list[0]) != 0:
                        sum_saliency[0] /= len(self._taken_picture_list[0])
                    sum_saliency = torch.tensor(sum_saliency, dtype=torch.float, device=current_episode_reward.device).unsqueeze(1)
                    current_episode_sum_saliency[0] += sum_saliency[0][0].item()
                            
                    # save description
                    out_path = os.path.join("output_descriptions/description.txt")
                    with open(out_path, 'a') as f:
                        # print関数でファイルに出力する
                        print(self.step,file=f)
                        print(pred_description,file=f)
                            
                    pbar.update()
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[0].item()
                    episode_stats["exp_area"] = current_episode_exp_area[0].item()
                    episode_stats["pic_sim"] = current_episode_picsim[0].item()
                    episode_stats["sum_saliency"] = current_episode_sum_saliency[0].item()
                                
                    episode_stats.update(
                        self._extract_scalars_from_info(infos)
                    )
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            #current_episodes[0].scene_id,
                            #current_episodes[0].episode_id,
                            "aaa", "aaa"
                        )
                    ] = episode_stats
                                
                    raw_metrics_episodes[
                        #current_episodes[0].scene_id + '.' + 
                        #current_episodes[0].episode_id
                        "aaa.aaa"
                    ] = infos["raw_metrics"]

                    if len(self.config.VIDEO_OPTION) > 0:
                        if len(rgb_frames[0]) == 0:
                            frame = observations_to_image(observations, infos, action.cpu().numpy())
                            rgb_frames[0].append(frame)
                        picture = rgb_frames[0][-1]
                        for j in range(20):
                            rgb_frames[0].append(picture) 
                        metrics=self._extract_scalars_from_info(infos)
                                
                        name_sim = str(len(stats_episodes)) + "-" + str(episode_stats["exp_area"])[:4] + "-" + str(self.step)
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR+"/"+date,
                            images=rgb_frames[0],
                            episode_id="aaa",
                            checkpoint_idx=checkpoint_index,
                            metrics=metrics,
                            name_ci=name_sim,
                        )
                        client.speak("途中経過のビデオを作成しました。")
                        
                        # Save taken picture                     
                        for j in range(len(self._taken_picture[i])):
                            picture_name = f"episode=aaa-{len(stats_episodes)}-{j}-{self.step}"
                            dir_name = "./taken_picture/" + date 
                            if not os.path.exists(dir_name):
                                os.makedirs(dir_name)
                                
                            picture = Image.fromarray(np.uint8(self._taken_picture[i][j]))
                            file_path = dir_name + "/" + picture_name + ".png"
                            picture.save(file_path)

            # episode ended
            pred_description = self.create_description(self._taken_picture_list[0])
            pic_sim[0] = self._calculate_pic_sim(self._taken_picture_list[0])
            current_episode_picsim[0] += pic_sim[0]

            for j in range(len(self._taken_picture_list[0])):
                sum_saliency[0] += self._taken_picture_list[0][j][0]
            if len(self._taken_picture_list[0]) != 0:
                sum_saliency[0] /= len(self._taken_picture_list[0])
            sum_saliency = torch.tensor(sum_saliency, dtype=torch.float, device=current_episode_reward.device).unsqueeze(1)
            current_episode_sum_saliency[0] += sum_saliency[0][0].item()
                    
            # save description
            out_path = os.path.join("output_descriptions/description.txt")
            with open(out_path, 'a') as f:
                # print関数でファイルに出力する
                print(pred_description,file=f)
                       
            pbar.update()
            episode_stats = dict()
            episode_stats["reward"] = current_episode_reward[0].item()
            episode_stats["exp_area"] = current_episode_exp_area[0].item()
            episode_stats["pic_sim"] = current_episode_picsim[0].item()
            episode_stats["sum_saliency"] = current_episode_sum_saliency[0].item()
                         
            episode_stats.update(
                self._extract_scalars_from_info(infos)
            )
            # use scene_id + episode_id as unique id for storing stats
            stats_episodes[
                (
                    #current_episodes[0].scene_id,
                    #current_episodes[0].episode_id,
                    "aaa", "aaa"
                )
            ] = episode_stats
                        
            raw_metrics_episodes[
                #current_episodes[0].scene_id + '.' + 
                #current_episodes[0].episode_id
                "aaa.aaa"
            ] = infos["raw_metrics"]

            if len(self.config.VIDEO_OPTION) > 0:
                if len(rgb_frames[0]) == 0:
                    frame = observations_to_image(observations, infos, action.cpu().numpy())
                    rgb_frames[0].append(frame)
                picture = rgb_frames[0][-1]
                for j in range(20):
                    rgb_frames[0].append(picture) 
                metrics=self._extract_scalars_from_info(infos)
                        
                name_sim = str(len(stats_episodes)) + "-" + str(episode_stats["exp_area"])[:4]
                generate_video(
                    video_option=self.config.VIDEO_OPTION,
                    video_dir=self.config.VIDEO_DIR+"/"+date,
                    images=rgb_frames[0],
                    episode_id="aaa",
                    checkpoint_idx=checkpoint_index,
                    metrics=metrics,
                    name_ci=name_sim,
                )
                client.speak("ビデオを作成しました。")
            
                # Save taken picture                     
                for j in range(len(self._taken_picture[i])):
                    picture_name = f"episode=aaa-{len(stats_episodes)}-{j}"
                    dir_name = "./taken_picture/" + date 
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                        
                    picture = Image.fromarray(np.uint8(self._taken_picture[i][j]))
                    file_path = dir_name + "/" + picture_name + ".png"
                    picture.save(file_path)
            
            client.speak("エピソードが終了しました。")
                    
            logger.info("Pic_Sim: " + str(episode_stats["pic_sim"]))
            logger.info("Sum Saliency: " + str(episode_stats["sum_saliency"]))

            self.env.close()
            client.return_shelf()
            client.return_home()
            break
            
