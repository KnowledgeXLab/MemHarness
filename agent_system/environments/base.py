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
from copy import deepcopy
import torch
import numpy as np
import os
from agent_system.environments.prompts import *
from collections import defaultdict
from agent_system.memory import MemoryManager

def to_numpy(data):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        pass
    elif isinstance(data, (int, float, bool, Tuple, List)):
        data = np.array(data)
    else:
        raise ValueError(f"Unsupported type: {type(data)})")
    return data

class EnvironmentManagerBase:
    def __init__(self, envs, projection_f, config):
        """
        Initialize the environment manager.
        
        Parameters:
        - envs: The environment instance, usually a vectorized environment containing multiple sub-environments.
        - projection_f: A function that maps text actions to environment actions.
        - config: Configuration object.
        """
        self.envs = envs
        self.projection_f = projection_f
        self.config = config
        self.retrieval_memory = MemoryManager(
            memory_config=self.config.env.memory,
            task_name=self.config.env.env_name,
        )

    def reset(self, kwargs) -> Dict[str, Any]:
        """
        Reset all environments and return the initial observations.
        Parameters:
        - kwargs (Dict): Additional keyword arguments for resetting the environment.

        Returns:
        - next_observations (Dict):
          - 'text' (None or List[str]): The textual observation.
          - 'image' (np.ndarray or torch.Tensor): The image observation as either a NumPy array or a PyTorch tensor.
          - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        """
        obs, infos = self.envs.reset()
        return {'text': None, 'image': obs, 'anchor': None}, infos
    
    def step(self, text_actions: List[str]):
        """
        Execute text actions and return the next state, rewards, done flags, and additional information.
        
        Parameters:
        - text_actions (List[str]): A list of text actions to execute.
        
        Returns:
        - next_observations (Dict):
          - 'text' (None or List[str]): The textual observation.
          - 'image' (np.ndarray or torch.Tensor): The image observation as either a NumPy array or a PyTorch tensor.
          - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        - rewards (np.ndarry or torch.Tensor): The rewards returned by the environment.
        - dones (np.ndarray or torch.Tensor): Done flags indicating which environments have completed.
        - infos (List[Dict]): Additional environment information.
        
        Exceptions:
        - NotImplementedError: If an observation key is not in ('text', 'image').
        """
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_observations = {
            'text': None, # Implement this if needed
            'image': next_obs,
            'anchor': None # For GiGPO only. anchor observation without any histories, hint, etc. Implement this if needed
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)
        
        return next_observations, rewards, dones, infos

    def build_text_obs(self,) -> List[str]:
        """
        This function builds the text observation for the agent.
        
        Returns:
        - postprocess_text_obs (List[str]): A list of processed text observations.
        """
        pass

    def close(self) -> None:
        """
        Close the environment and release resources.
        """
        if self.retrieval_memory is not None:
            self.retrieval_memory.close()
        self.envs.close()

    def _memory_enabled(self) -> bool:
        return self.retrieval_memory is not None and self.retrieval_memory.enabled

    def _memory_retrieval_mode(self) -> str:
        if not self._memory_enabled():
            return "disabled"
        return self.retrieval_memory.retrieval_mode()

    def _finalize_text_prompt_with_memory(self, prompt: str, state_text: str) -> str:
        if not self._memory_enabled():
            return prompt

        if self.retrieval_memory.is_fixed_mode():
            memory_prompt = self.retrieval_memory.build_memory_message(state_text=state_text)
            return self._append_external_memory(prompt, memory_prompt)

        if self.retrieval_memory.is_agentic_mode():
            return self.retrieval_memory.append_retrieval_hint(prompt)

        return prompt

    def _extract_memory_queries(self, text_actions: List[str]) -> List[str | None]:
        if not self._memory_enabled() or not self.retrieval_memory.is_agentic_mode():
            return [None] * len(text_actions)
        return [self.retrieval_memory.extract_query(action) for action in text_actions]

    def _inject_memory_queries_into_observations(
        self,
        observations: Dict[str, Any],
        query_texts: List[str | None],
    ) -> tuple[Dict[str, Any], List[dict | None], List[str]]:
        cloned_observations = {
            key: (list(value) if isinstance(value, list) else deepcopy(value))
            for key, value in observations.items()
        }
        if not self._memory_enabled():
            return cloned_observations, [None] * len(query_texts), [""] * len(query_texts)

        texts = list(cloned_observations.get("text") or [])
        anchors = cloned_observations.get("anchor") or texts
        serialized_events: List[dict | None] = [None] * len(query_texts)
        injected_texts: List[str] = [""] * len(query_texts)
        for idx, query_text in enumerate(query_texts):
            if not query_text:
                continue
            state_text = anchors[idx] if idx < len(anchors) else texts[idx]
            injected_text, event = self.retrieval_memory.build_memory_message_and_event(
                state_text=state_text,
                query_text=query_text,
            )
            texts[idx] = self._append_external_memory(texts[idx], injected_text)
            injected_texts[idx] = injected_text
            serialized_events[idx] = event.to_dict() if event is not None else None
        cloned_observations["text"] = texts
        return cloned_observations, serialized_events, injected_texts

    def step_with_memory(
        self,
        text_actions: List[str],
        current_obs: Dict[str, Any] | None = None,
        current_infos: List[Dict[str, Any]] | None = None,
    ):
        del current_obs, current_infos
        if self._memory_retrieval_mode() == "agentic":
            raise NotImplementedError(f"{self.__class__.__name__} does not support agentic memory retrieval yet.")
        next_obs, rewards, dones, infos = self.step(text_actions)
        for info in infos:
            info["memory_action_type"] = "env_action"
            info["memory_query_text"] = None
            info["memory_injected_text"] = ""
            info["memory_event"] = None
        env_step_mask = np.ones(len(text_actions), dtype=bool)
        return next_obs, rewards, dones, infos, env_step_mask

    def maybe_write_rollout_memories(self, **kwargs):
        if self.retrieval_memory is None:
            return []
        return self.retrieval_memory.maybe_write_rollout_memories(**kwargs)

    @staticmethod
    def _append_external_memory(prompt: str, memory_prompt: str) -> str:
        if not memory_prompt:
            return prompt
        return f"{prompt}\n\n{memory_prompt}"

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Evaluate if the episodes are successful or not. 
        (Default) implementation is to check info['won'] of the last step.
        
        Returns:
        - success (np.ndarray or torch.Tensor): 1 if the episode is successful, 0 otherwise.
        """
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        batch_size = len(total_batch_list)
        
        success = defaultdict(list)
        
        for bs in range(batch_size):
            self._process_batch(bs, total_batch_list, total_infos, success)
        
        assert len(success['success_rate']) == batch_size

        return {key: np.array(value) for key, value in success.items()}

    def _get_latest_effective_step(self, batch_idx, total_batch_list, total_infos):
        fallback = None
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if not batch_item['active_masks']:
                continue

            info = total_infos[batch_idx][i]
            if fallback is None:
                fallback = (batch_item, info)

            memory_action_type = info.get("memory_action_type", "env_action")
            env_step_mask = bool(batch_item.get("env_step_mask", True))
            if memory_action_type != "memory_query" or env_step_mask:
                return batch_item, info

        return fallback
    
    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        latest_step = self._get_latest_effective_step(batch_idx, total_batch_list, total_infos)
        if latest_step is None:
            success['success_rate'].append(0.0)
            return

        _, info = latest_step
        won_value = float(info.get('won', False))
        success['success_rate'].append(won_value)
            
    def save_image(self, image, step):
        """
        Save an image to a file.
        
        Parameters:
        - image (np.ndarray or torch.Tensor): The image to save.
        - path (str): The path to save the image.
        """
        path = os.path.join(os.path.dirname(__file__), os.path.join("images", self.config.env.env_name))
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, f"step{step}.png")
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError(f"Unsupported type: {type(image)})")
        
        if len(image.shape) == 4:
            image = image[0]
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        if image.max() <= 1.0:
            image = (image * 255)

        image = image.astype(np.uint8)
        
        from PIL import Image
        image = Image.fromarray(image)
        image.save(path)