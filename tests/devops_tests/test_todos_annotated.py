""" utilities to ensure all TO-DO comments in the code are annotated
    with an id of an open GitHub issue """
import os
import re
import sys
import pathlib
import warnings
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    from fastcore.net import ExceptionsHTTP

from ghapi.all import GhApi, paged

# pylint: disable-next=redefined-outer-name
def test_todos_annotated(project_file, gh_issues):
    """ pytest test reporting status of each file in the project """
    if (
        os.path.basename(project_file) == 'test_todos_annotated.py' or
        project_file.endswith("-checkpoint.ipynb") or
        ".eggs" in project_file
    ):
        return
    for line in _grep(project_file, r'.*TODO.*'):
        match = re.search(r'TODO #(\d+)', line)
        if match is None:
            raise Exception(f"TODO not annotated with issue id ({line})")
        giving_up_with_hope_other_builds_did_it = len(gh_issues) == 0
        if not giving_up_with_hope_other_builds_did_it:
            number = int(match.group(1))
            if number not in gh_issues.keys():
                raise Exception(f"TODO annotated with non-existent id ({line})")
            if gh_issues[number] != 'open':
                raise Exception(f"TODO remains for a non-open issue ({line})")


# https://stackoverflow.com/questions/7012921/recursive-grep-using-python
def _findfiles(path, regex):
    reg_obj = re.compile(regex)
    res = []
    for root, _, fnames in os.walk(path):
        for fname in fnames:
            if reg_obj.match(fname):
                res.append(os.path.join(root, fname))
    return res


def _grep(filepath, regex):
    reg_obj = re.compile(regex)
    res = []
    with open(filepath, encoding="utf8") as file:
        for line in file:
            if reg_obj.match(line):
                res.append(line)
    return res


@pytest.fixture(
    params=_findfiles(
        pathlib.Path(__file__).parent.parent.parent.absolute(),
        r'.*\.(ipynb|py|txt|yml|m|jl|md)$'
    )
)
def project_file(request):
    """ pytest fixture enabling execution of the test for each project file """
    return request.param


@pytest.fixture(scope='session')
def gh_issues():
    """ pytest fixture providing a dictionary with github issue ids as keys
        and their state as value """
    res = {}
    if 'CI' not in os.environ or ('GITHUB_ACTIONS' in os.environ and sys.version_info.minor >= 8):
        try:
            api = GhApi(owner='atmos-cloud-sim-uj', repo='PyMPDATA')
            pages = paged(
                api.issues.list_for_repo,
                owner='atmos-cloud-sim-uj',
                repo='PyMPDATA',
                state='all',
                per_page=100
            )
            for page in pages:
                for item in page.items:
                    res[item.number] = item.state
        except ExceptionsHTTP[403]:
            pass
    return res
