import os
import sys
import shutil
import tempfile
import argparse
import urllib.request
import zipfile

_CITATIONS = """
@article{wang2018glue,
  title={Glue: A multi-task benchmark and analysis platform for natural language understanding},
  author={Wang, Alex and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel R},
  journal={arXiv preprint arXiv:1804.07461},
  year={2018}
}

@inproceedings{wang2019superglue,
  title={Superglue: A stickier benchmark for general-purpose language understanding systems},
  author={Wang, Alex and Pruksachatkun, Yada and Nangia, Nikita and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel},
  booktitle={Advances in Neural Information Processing Systems},
  pages={3261--3275},
  year={2019}
}
"""

GLUE_TASKS = ["CoLA", "SST", "MRPC", "QQP", "STS", "MNLI", "SNLI", "QNLI", "RTE", "WNLI",
              "diagnostic"]
SUPERGLUE_TASKS = ["CB", "COPA", "MultiRC", "RTE", "WiC", "WSC", "BoolQ", "ReCoRD",
                   "diagnostic"]


GLUE_TASK2PATH = {
    "CoLA": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4",  # noqa
    "SST": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8",  # noqa
    "MRPC": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc",  # noqa
    "QQP": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQQP-clean.zip?alt=media&token=11a647cb-ecd3-49c9-9d31-79f8ca8fe277",  # noqa
    "STS": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSTS-B.zip?alt=media&token=bddb94a7-8706-4e0d-a694-1109e12273b5",  # noqa
    "MNLI": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce",  # noqa
    "SNLI": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSNLI.zip?alt=media&token=4afcfbb2-ff0c-4b2d-a09a-dbf07926f4df",  # noqa
    "QNLI": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQNLIv2.zip?alt=media&token=6fdcf570-0fc5-4631-8456-9505272d1601",  # noqa
    "RTE": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FRTE.zip?alt=media&token=5efa7e85-a0bb-4f19-8ea2-9e1840f077fb",  # noqa
    "WNLI": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FWNLI.zip?alt=media&token=068ad0a0-ded7-4bd7-99a5-5e00222e0faf",  # noqa
    "diagnostic": [
        "https://storage.googleapis.com/mtl-sentence-representations.appspot.com/tsvsWithoutLabels%2FAX.tsv?GoogleAccessId=firebase-adminsdk-0khhl@mtl-sentence-representations.iam.gserviceaccount.com&Expires=2498860800&Signature=DuQ2CSPt2Yfre0C%2BiISrVYrIFaZH1Lc7hBVZDD4ZyR7fZYOMNOUGpi8QxBmTNOrNPjR3z1cggo7WXFfrgECP6FBJSsURv8Ybrue8Ypt%2FTPxbuJ0Xc2FhDi%2BarnecCBFO77RSbfuz%2Bs95hRrYhTnByqu3U%2FYZPaj3tZt5QdfpH2IUROY8LiBXoXS46LE%2FgOQc%2FKN%2BA9SoscRDYsnxHfG0IjXGwHN%2Bf88q6hOmAxeNPx6moDulUF6XMUAaXCSFU%2BnRO2RDL9CapWxj%2BDl7syNyHhB7987hZ80B%2FwFkQ3MEs8auvt5XW1%2Bd4aCU7ytgM69r8JDCwibfhZxpaa4gd50QXQ%3D%3D",  # noqa
        "https://www.dropbox.com/s/ju7d95ifb072q9f/diagnostic-full.tsv?dl=1",
    ],
}


SUPERGLUE_TASK2PATH = {
    "CB": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/CB.zip",
    "COPA": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/COPA.zip",
    "MultiRC": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/MultiRC.zip",
    "RTE": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/RTE.zip",
    "WiC": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WiC.zip",
    "WSC": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WSC.zip",
    "broadcoverage-diagnostic": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/AX-b.zip",
    "winogender-diagnostic": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/AX-g.zip",
    "BoolQ": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip",
    "ReCoRD": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/ReCoRD.zip",
}


def download_and_extract(task, data_dir):
    print("Downloading and extracting %s..." % task)
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    data_file = os.path.join(data_dir, "%s.zip" % task)
    urllib.request.urlretrieve(TASK2PATH[task], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)
    print(f"\tCompleted! Downloaded {task} data to directory {data_dir}")


def download_diagnostic(data_dir):
    print("Downloading and extracting diagnostic...")
    if not os.path.isdir(os.path.join(data_dir, "RTE")):
        os.mkdir(os.path.join(data_dir, "RTE"))
    diagnostic_dir = os.path.join(data_dir, "RTE", "diagnostics")
    if not os.path.isdir(diagnostic_dir):
        os.mkdir(diagnostic_dir)

    data_file = os.path.join(diagnostic_dir, "broadcoverage.zip")
    urllib.request.urlretrieve(TASK2PATH["broadcoverage-diagnostic"], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(diagnostic_dir)
        shutil.move(os.path.join(diagnostic_dir, "AX-b", "AX-b.jsonl"), diagnostic_dir)
        os.rmdir(os.path.join(diagnostic_dir, "AX-b"))
    os.remove(data_file)

    data_file = os.path.join(diagnostic_dir, "winogender.zip")
    urllib.request.urlretrieve(TASK2PATH["winogender-diagnostic"], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(diagnostic_dir)
        shutil.move(os.path.join(diagnostic_dir, "AX-g", "AX-g.jsonl"), diagnostic_dir)
        os.rmdir(os.path.join(diagnostic_dir, "AX-g"))
    os.remove(data_file)
    print("\tCompleted!")
    return


def get_tasks(task_names):
    task_names = task_names.split(",")
    if "all" in task_names:
        tasks = TASKS
    else:
        tasks = []
        for task_name in task_names:
            assert task_name in TASKS, "Task %s not found!" % task_name
            tasks.append(task_name)
        if "RTE" in tasks and "diagnostic" not in tasks:
            tasks.append("diagnostic")
    return tasks


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_dir", help="directory to save data to", type=str, default="../superglue_data"
    )
    parser.add_argument(
        "-t",
        "--tasks",
        help="tasks to download data for as a comma separated string",
        type=str,
        default="all",
    )
    args = parser.parse_args(arguments)

    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
    tasks = get_tasks(args.tasks)

    for task in tasks:
        if task == "diagnostic":
            download_diagnostic(args.data_dir)
        else:
            download_and_extract(task, args.data_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
