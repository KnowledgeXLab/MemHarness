# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Shared memory prompt defaults for runtime (agentic retrieval hints) and data scripts.

# Matches verl/trainer/config/ppo_trainer.yaml env.memory.retrieval_instruction_prompt
DEFAULT_RETRIEVAL_INSTRUCTION_PROMPT = (
    "If you want to retrieve memory to get some guidance before acting, output exactly one query "
    "wrapped by {open_tag} and {close_tag}. After memory is returned, you will act in the next response."
)

# Stronger hint for general / API models without memory-specific SFT (see examples/grpo_trainer/*-test.sh).
GENERAL_MODEL_RETRIEVAL_INSTRUCTION_PROMPT = (
    "You have access to a memory bank of past experiences from similar tasks. "
    "When you are uncertain how to proceed, prefer retrieving relevant memory for guidance before "
    "taking an environment action. To retrieve memory, output exactly one search query wrapped by "
    "{open_tag} and {close_tag}. Do not output both a memory retrieval query and an action in the "
    "same response. After memory is returned, you will act in the next response."
)
