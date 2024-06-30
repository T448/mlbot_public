from typing import List
import git
from common.imported_modules import get_imported_modules

IGNORE_FILES = set(["git_utils", "imported_modules"])


def has_diff(hash: str, file_list: List[str]) -> bool:
    repo = git.Repo("/home/runner/mlbot2/")
    try:
        diffs: List[str] = repo.git.diff(hash).split("\n")
        print("repo.git.diff(hash)", repo.git.diff(hash))
    except Exception:
        return False

    diffs = [i.replace("+++ b", "") for i in diffs if ("+++ b" in i) and ("src" in i)]

    print("hash : ", hash)
    print("diffs : ", diffs)
    print("get_imported_modules:", get_imported_modules())
    print("file_list : ", file_list)

    file_list = [i for i in file_list if i not in IGNORE_FILES]

    for imported_module in file_list:
        for diff in diffs:
            if diff in imported_module:
                if sum([ignore_file in diff for ignore_file in IGNORE_FILES]) == 0:
                    return True
    return False
