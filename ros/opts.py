import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_dir',
        default='/media/minjee/4970a4b3-9bec-42aa-8022-ddff6e7b8f80',
        type=str,
        help='Root Directory path')
    parser.add_argument(
        '--bag_dir',
        default='/media/minjee/4970a4b3-9bec-42aa-8022-ddff6e7b8f80/bagfiles',
        type=str,
        help='Directory path of Bag files')
    parser.add_argument(
        '--img_dir',
        default='/media/minjee/4970a4b3-9bec-42aa-8022-ddff6e7b8f80/images',
        type=str,
        help='Directory path of Images to be stored')
    parser.add_argument(
        '--label_dir',
        default='/media/minjee/4970a4b3-9bec-42aa-8022-ddff6e7b8f80/labeled',
        type=str,
        help='Directory path of Images to be labeled')
    parser.add_argument(
        '--excel_path',
        default='/data/projects/handhygiene/data/hh_label.xlsx',
        type=str,
        help='Annotation file path')

    args = parser.parse_args()

    return args