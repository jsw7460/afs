from typing import Dict, Tuple

from .ra_networks import LSTMSubTrajectoryBasedSkillGenerator, RAMLPSkillPrior, RALowLevelSkillPolicy, RALowLevelSkillPolicyLogStd
from .sopt_skill_prior import SOPTSkillEmpowered


class RASkillPrior(SOPTSkillEmpowered):
    def __init__(self, *args, **kwargs):
        super(RASkillPrior, self).__init__(*args, **kwargs)

    @property
    def skill_generator_class(self):
        return LSTMSubTrajectoryBasedSkillGenerator

    @property
    def skill_prior_class(self):
        return RAMLPSkillPrior

    @property
    def skill_decoder_class(self):
        return RALowLevelSkillPolicy

    def build_hrl_models(self, hrl_config: Dict) -> Tuple[int, int]:
        raise NotImplementedError()
