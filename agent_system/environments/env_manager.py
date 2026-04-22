# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
import ray
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory
from agent_system.environments.env_package.alfworld.envs import compute_reward
from omegaconf import OmegaConf

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))

        mem_ev: list[dict | None] = []
        observations = {
            "text": self.build_text_obs(obs, init=True, memory_events_out=mem_ev),
            "image": None,
            "anchor": obs.copy()
        }
        for i, info in enumerate(infos):
            info["memory_event"] = mem_ev[i] if i < len(mem_ev) else None
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        mem_ev: list[dict | None] = []
        next_observations = {
            "text": self.build_text_obs(next_obs, memory_events_out=mem_ev),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])
            info["memory_event"] = mem_ev[i] if i < len(mem_ev) else None

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False,
        memory_events_out: list[dict | None] | None = None,
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            obs_i, ev = self._pair_finalize_memory(obs_i, text_obs[i])
            postprocess_text_obs.append(obs_i)
            if memory_events_out is not None:
                memory_events_out.append(ev.to_dict() if ev is not None else None)

        return postprocess_text_obs

    def step_with_memory(self, text_actions: List[str], current_obs=None, current_infos=None):
        # Not agentic memory retrieval mode, skip memory retrieval
        if self._memory_retrieval_mode() != "agentic":
            return super().step_with_memory(text_actions, current_obs=current_obs, current_infos=current_infos)

        # Extract memory queries from text actions
        query_texts = self._extract_memory_queries(text_actions)

        # No memory-query tokens: run a normal env step. Do not call base step_with_memory (agentic mode raises NotImplementedError there).
        if not any(query_texts):
            next_obs, rewards, dones, infos = self.step(text_actions)
            for info in infos:
                info["memory_action_type"] = "env_action"
                info["memory_query_text"] = None
                info["memory_injected_text"] = ""
                info["memory_event"] = None
            env_step_mask = np.ones(len(text_actions), dtype=bool)
            return next_obs, rewards, dones, infos, env_step_mask

        actions, valids = self.projection_f(text_actions)
        current_raw_obs = list((current_obs or {}).get("anchor", []))
        next_raw_obs = list(current_raw_obs)
        rewards = np.zeros(len(text_actions), dtype=np.float32)
        dones = np.zeros(len(text_actions), dtype=bool)
        infos: List[Dict[str, Any]] = []
        env_step_mask = np.zeros(len(text_actions), dtype=bool)

        for idx, query_text in enumerate(query_texts):
            if query_text:
                infos.append(
                    {
                        "is_action_valid": True,
                        "memory_action_type": "memory_query",
                        "memory_query_text": query_text,
                        "memory_injected_text": "",
                        "memory_event": None,
                        "won": False,  # placeholder; not used for Search
                    }
                )
                continue

            obs, reward, done, info = self.envs._sync_step(self.envs.envs[idx], actions[idx])
            next_raw_obs[idx] = obs
            self.memory[idx].append({"search": actions[idx], "information": obs})
            info["is_action_valid"] = to_numpy(valids[idx])
            info["memory_action_type"] = "env_action"
            info["memory_query_text"] = None
            info["memory_injected_text"] = ""
            info["memory_event"] = None
            rewards[idx] = reward
            dones[idx] = done
            env_step_mask[idx] = True
            infos.append(info)

        next_observations = {
            "text": self.build_text_obs(next_raw_obs),
            "image": None,
            "anchor": next_raw_obs.copy(),
        }
        next_observations, memory_events, injected_texts = self._inject_memory_queries_into_observations(
            next_observations,
            query_texts,
        )
        for idx, query_text in enumerate(query_texts):
            if query_text:
                infos[idx]["memory_injected_text"] = injected_texts[idx]
                infos[idx]["memory_event"] = memory_events[idx]

        return next_observations, rewards, dones, infos, env_step_mask


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        latest_step = self._get_latest_effective_step(batch_idx, total_batch_list, total_infos)
        if latest_step is None:
            success['success_rate'].append(0.0)
            success["unknown_success_rate"].append(0.0)
            return

        _, info = latest_step
        won_value = float(info.get('won', False))
        success['success_rate'].append(won_value)

        data_source = info.get("data_source", "unknown")
        success[f"{data_source}_success_rate"].append(won_value)
        return
            

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        mem_ev: list[dict | None] = []
        full_text_obs = self.build_text_obs(
            text_obs, self.envs.get_admissible_commands, init=True, memory_events_out=mem_ev
        )
        for i, info in enumerate(infos):
            info["memory_event"] = mem_ev[i] if i < len(mem_ev) else None
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        mem_ev: list[dict | None] = []
        full_text_obs = self.build_text_obs(
            text_obs, self.envs.get_admissible_commands, memory_events_out=mem_ev
        )
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info["memory_event"] = mem_ev[i] if i < len(mem_ev) else None

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(
        self,
        text_obs: List[str],
        admissible_actions: List[List[str]],
        init: bool = False,
        memory_events_out: list[dict | None] | None = None,
    ) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            obs, ev = self._pair_finalize_memory(obs, text_obs[i])
            postprocess_text_obs.append(obs)
            if memory_events_out is not None:
                memory_events_out.append(ev.to_dict() if ev is not None else None)
        return postprocess_text_obs

    def step_with_memory(self, text_actions: List[str], current_obs=None, current_infos=None):
        del current_infos
        if self._memory_retrieval_mode() != "agentic":
            return super().step_with_memory(text_actions, current_obs=current_obs)

        query_texts = self._extract_memory_queries(text_actions)
        if not any(query_texts):
            # All env actions this step (no memory query): do not call base step_with_memory
            # (it raises NotImplementedError in agentic mode). Match base non-agentic behavior.
            next_obs, rewards, dones, infos = self.step(text_actions)
            for info in infos:
                info["memory_action_type"] = "env_action"
                info["memory_query_text"] = None
                info["memory_injected_text"] = ""
                info["memory_event"] = None
            env_step_mask = np.ones(len(text_actions), dtype=bool)
            return next_obs, rewards, dones, infos, env_step_mask

        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        next_text_obs = list(self.pre_text_obs)
        rewards = np.zeros(len(text_actions), dtype=np.float32)
        dones = np.zeros(len(text_actions), dtype=bool)
        infos: List[Dict[str, Any]] = []
        env_step_mask = np.zeros(len(text_actions), dtype=bool)

        # Parallel env steps: submit all worker.step.remote first, one ray.get (same pattern as
        # AlfworldEnvs.step). Avoids O(batch) serial latency when some slots are memory_query.
        step_indices = [idx for idx, q in enumerate(query_texts) if not q]
        step_results: Dict[int, Tuple[Any, Any, Any, Dict[str, Any]]] = {}
        if step_indices:
            futures = [
                self.envs.workers[idx].step.remote(actions[idx]) for idx in step_indices
            ]
            raw = ray.get(futures)
            for j, idx in enumerate(step_indices):
                obs, scores, done_ls, info = raw[j]
                for key in info.keys():
                    info[key] = info[key][0]
                step_results[idx] = (obs, scores, done_ls, info)

        for idx, query_text in enumerate(query_texts):
            if query_text:
                infos.append(
                    {
                        "is_action_valid": True,
                        "memory_action_type": "memory_query",
                        "memory_query_text": query_text,
                        "memory_injected_text": "",
                        "memory_event": None,
                        "won": False,
                    }
                )
                continue

            obs, scores, done_ls, info = step_results[idx]
            next_text_obs[idx] = obs[0]
            self.memory[idx].append({"text_obs": self.pre_text_obs[idx], "action": actions[idx]})
            self.pre_text_obs[idx] = obs[0]
            rewards[idx] = compute_reward(info, self.envs.multi_modal)
            dones[idx] = done_ls[0]
            env_step_mask[idx] = True
            info["is_action_valid"] = to_numpy(valids[idx])
            info["memory_action_type"] = "env_action"
            info["memory_query_text"] = None
            info["memory_injected_text"] = ""
            info["memory_event"] = None
            infos.append(info)
            # Keep admissible_commands cache in sync with AlfworldEnvs.step (used by get_admissible_commands).
            if "admissible_commands" in info:
                self.envs.prev_admissible_commands[idx] = info["admissible_commands"]

        infos = set_gamefile(infos, self.gamefile)
        image_obs = self.envs.getobs() if self.envs.multi_modal else None
        next_observations = {
            "text": self.build_text_obs(next_text_obs, self.envs.get_admissible_commands),
            "image": image_obs,
            "anchor": next_text_obs,
        }
        next_observations, memory_events, injected_texts = self._inject_memory_queries_into_observations(
            next_observations,
            query_texts,
        )
        for idx, query_text in enumerate(query_texts):
            if query_text:
                infos[idx]["memory_injected_text"] = injected_texts[idx]
                infos[idx]["memory_event"] = memory_events[idx]

        return next_observations, rewards, dones, infos, env_step_mask

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        latest_step = self._get_latest_effective_step(batch_idx, total_batch_list, total_infos)
        if latest_step is None:
            success['success_rate'].append(0.0)
            return

        _, info = latest_step
        won_value = float(info.get('won', False))
        success['success_rate'].append(won_value)

        gamefile = info.get("extra.gamefile")
        if gamefile:
            self._process_gamefile(gamefile, won_value, success)
        return

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        mem_ev: list[dict | None] = []
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True, memory_events_out=mem_ev),
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True, memory_events_out=mem_ev),
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        for i, info in enumerate(infos):
            info["memory_event"] = mem_ev[i] if i < len(mem_ev) else None
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        mem_ev: list[dict | None] = []
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos, memory_events_out=mem_ev),
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs, memory_events_out=mem_ev),
                'image': None, 
                'anchor': next_obs 
            }

        for i, info in enumerate(infos):
            info["memory_event"] = mem_ev[i] if i < len(mem_ev) else None

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        infos,
        text_obs: List[str] = None,
        init: bool = False,
        memory_events_out: list[dict | None] | None = None,
    ) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            ev_dict = None
            if text_obs is not None:
                obs, ev = self._pair_finalize_memory(obs, text_obs[i])
                ev_dict = ev.to_dict() if ev is not None else None
            postprocess_text_obs.append(obs)
            if memory_events_out is not None:
                memory_events_out.append(ev_dict)

        return postprocess_text_obs


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        mem_ev: list[dict | None] = []
        observations = {'text': self.build_text_obs(obs, infos, init=True, memory_events_out=mem_ev), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        for i, info in enumerate(infos):
            info["memory_event"] = mem_ev[i] if i < len(mem_ev) else None
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        mem_ev: list[dict | None] = []
        next_observations = {
            'text': self.build_text_obs(next_obs, infos, memory_events_out=mem_ev),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info["memory_event"] = mem_ev[i] if i < len(mem_ev) else None

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(
        self,
        text_obs: List[str],
        infos: List[List[str]],
        init: bool = False,
        memory_events_out: list[dict | None] | None = None,
    ) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            obs, ev = self._pair_finalize_memory(obs, text_obs[i])
            postprocess_text_obs.append(obs)
            if memory_events_out is not None:
                memory_events_out.append(ev.to_dict() if ev is not None else None)

        return postprocess_text_obs

    def step_with_memory(self, text_actions: List[str], current_obs=None, current_infos=None):
        if self._memory_retrieval_mode() != "agentic":
            return super().step_with_memory(text_actions, current_obs=current_obs, current_infos=current_infos)

        query_texts = self._extract_memory_queries(text_actions)
        if not any(query_texts):
            next_obs, rewards, dones, infos = self.step(text_actions)
            for info in infos:
                info["memory_action_type"] = "env_action"
                info["memory_query_text"] = None
                info["memory_injected_text"] = ""
                info["memory_event"] = None
            env_step_mask = np.ones(len(text_actions), dtype=bool)
            return next_obs, rewards, dones, infos, env_step_mask

        actions, valids = self.projection_f(text_actions)
        next_raw_obs = list(self.pre_text_obs)
        infos: List[Dict[str, Any]] = []
        rewards = np.zeros(len(text_actions), dtype=np.float32)
        dones = np.zeros(len(text_actions), dtype=bool)
        env_step_mask = np.zeros(len(text_actions), dtype=bool)
        base_infos = list(current_infos or [{} for _ in range(len(text_actions))])

        for idx, query_text in enumerate(query_texts):
            if query_text:
                query_info = dict(base_infos[idx])
                query_info["is_action_valid"] = True
                query_info["memory_action_type"] = "memory_query"
                query_info["memory_query_text"] = query_text
                query_info["memory_injected_text"] = ""
                query_info["memory_event"] = None
                infos.append(query_info)
                continue

            obs, reward, done, info = ray.get(self.envs._workers[idx].step.remote(actions[idx]))
            formatted_obs = self.format_obs([obs])[0]
            next_raw_obs[idx] = formatted_obs
            self.memory[idx].append({"text_obs": self.pre_text_obs[idx], "action": actions[idx]})
            self.pre_text_obs[idx] = formatted_obs
            info["is_action_valid"] = to_numpy(valids[idx])
            info["memory_action_type"] = "env_action"
            info["memory_query_text"] = None
            info["memory_injected_text"] = ""
            info["memory_event"] = None
            rewards[idx] = reward
            dones[idx] = done
            env_step_mask[idx] = True
            infos.append(info)

        next_observations = {
            "text": self.build_text_obs(next_raw_obs, infos),
            "image": None,
            "anchor": next_raw_obs.copy(),
        }
        next_observations, memory_events, injected_texts = self._inject_memory_queries_into_observations(
            next_observations,
            query_texts,
        )
        for idx, query_text in enumerate(query_texts):
            if query_text:
                infos[idx]["memory_injected_text"] = injected_texts[idx]
                infos[idx]["memory_event"] = memory_events[idx]

        return next_observations, rewards, dones, infos, env_step_mask

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        latest_step = self._get_latest_effective_step(batch_idx, total_batch_list, total_infos)
        if latest_step is None:
            success['success_rate'].append(0.0)
            success['webshop_task_score (not success_rate)'].append(0.0)
            return

        _, info = latest_step
        won_value = float(info.get('won', False))
        score_value = float(info.get('task_score', 0.0))
        success['success_rate'].append(won_value)
        success['webshop_task_score (not success_rate)'].append(score_value)
        return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        mem_ev: list[dict | None] = []
        full_text_obs = self.build_text_obs(text_obs, init=True, memory_events_out=mem_ev)
        for i, info in enumerate(infos):
            info["memory_event"] = mem_ev[i] if i < len(mem_ev) else None
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        mem_ev: list[dict | None] = []
        full_text_obs = self.build_text_obs(text_obs, memory_events_out=mem_ev)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info["memory_event"] = mem_ev[i] if i < len(mem_ev) else None

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False,
        memory_events_out: list[dict | None] | None = None,
    ) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                obs, ev = self._pair_finalize_memory(obs, text_obs[i])
                postprocess_text_obs.append(obs)
                if memory_events_out is not None:
                    memory_events_out.append(ev.to_dict() if ev is not None else None)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                obs, ev = self._pair_finalize_memory(obs, text_obs[i])
                postprocess_text_obs.append(obs)
                if memory_events_out is not None:
                    memory_events_out.append(ev.to_dict() if ev is not None else None)
        return postprocess_text_obs

    def step_with_memory(self, text_actions: List[str], current_obs=None, current_infos=None):
        del current_infos
        if self._memory_retrieval_mode() != "agentic":
            return super().step_with_memory(text_actions, current_obs=current_obs)

        query_texts = self._extract_memory_queries(text_actions)
        if not any(query_texts):
            next_obs, rewards, dones, infos = self.step(text_actions)
            for info in infos:
                info["memory_action_type"] = "env_action"
                info["memory_query_text"] = None
                info["memory_injected_text"] = ""
                info["memory_event"] = None
            env_step_mask = np.ones(len(text_actions), dtype=bool)
            return next_obs, rewards, dones, infos, env_step_mask

        actions, valids = self.projection_f(text_actions)
        next_text_obs = list(self.pre_text_obs)
        rewards = np.zeros(len(text_actions), dtype=np.float32)
        dones = np.zeros(len(text_actions), dtype=bool)
        infos: List[Dict[str, Any]] = []
        env_step_mask = np.zeros(len(text_actions), dtype=bool)

        for idx, query_text in enumerate(query_texts):
            if query_text:
                infos.append(
                    {
                        "is_action_valid": True,
                        "memory_action_type": "memory_query",
                        "memory_query_text": query_text,
                        "memory_injected_text": "",
                        "memory_event": None,
                        "won": False,
                    }
                )
                continue

            obs, reward, done, info = ray.get(self.envs.workers[idx].step.remote(actions[idx]))
            next_text_obs[idx] = obs
            self.memory[idx].append({"text_obs": self.pre_text_obs[idx], "action": actions[idx]})
            self.pre_text_obs[idx] = obs
            info["is_action_valid"] = to_numpy(valids[idx])
            info["memory_action_type"] = "env_action"
            info["memory_query_text"] = None
            info["memory_injected_text"] = ""
            info["memory_event"] = None
            rewards[idx] = reward
            dones[idx] = done
            env_step_mask[idx] = True
            infos.append(info)

        next_observations = {
            "text": self.build_text_obs(next_text_obs),
            "image": None,
            "anchor": next_text_obs,
        }
        next_observations, memory_events, injected_texts = self._inject_memory_queries_into_observations(
            next_observations,
            query_texts,
        )
        for idx, query_text in enumerate(query_texts):
            if query_text:
                infos[idx]["memory_injected_text"] = injected_texts[idx]
                infos[idx]["memory_event"] = memory_events[idx]

        return next_observations, rewards, dones, infos, env_step_mask

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _val_on_train = bool(config.env.alfworld.get("validate_on_train_split", False))
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_alfworld_envs(
            alf_config_path,
            config.env.seed + 1000,
            config.data.val_batch_size,
            1,
            is_train=_val_on_train,
            env_kwargs=env_kwargs,
            resources_per_worker=resources_per_worker,
        )
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0, resources_per_worker=resources_per_worker)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)