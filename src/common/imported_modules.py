import distutils

SEARCH_TARGET_DIRS = ["entity.", "backtest.", "logic.", "common."]


def get_imported_modules():
    return [
        str(j).split("from '")[1].split("'>")[0]
        for i, j in distutils.sys.modules.items()
        if (sum([(search_target_dir in i) for search_target_dir in SEARCH_TARGET_DIRS]) > 0)
        and ("from '" in str(j))
        and ("'>" in str(j))
    ]
