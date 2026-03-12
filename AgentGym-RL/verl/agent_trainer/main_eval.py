# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

import hydra
from verl.utils.fs import copy_local_path_from_hdfs
import pandas as pd
import numpy as np


@hydra.main(config_path='config', config_name='evaluation', version_base=None)
def main(config):
    local_path = copy_local_path_from_hdfs(config.data.path)
    dataset = pd.read_json(local_path)
    reward_model_data = dataset[config.data.reward_model_key]

    passes = 0
    avgs = 0

    total = len(dataset)

    for i in range(total):
        passes += np.max(reward_model_data[i])
        avgs += np.mean(reward_model_data[i])

    print(f'pass@{len(reward_model_data[0])}: {passes / total}')
    print(f'avg@{len(reward_model_data[0])}: {avgs / total}')


if __name__ == '__main__':
    main()
