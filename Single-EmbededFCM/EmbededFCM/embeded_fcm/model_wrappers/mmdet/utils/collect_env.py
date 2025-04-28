# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash

from ..version import __version__ as mmdet_version


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['mmdetection'] = mmdet_version + '+' + get_git_hash()[:7]
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
