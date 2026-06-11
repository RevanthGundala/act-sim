from configs.act.base import ACTConfig
from utils.config import TRAIN_DATASET_REPO_ID


config = ACTConfig(
    dataset_repo_id=TRAIN_DATASET_REPO_ID,
    job_name="act_pour_water_wrist_top",
)