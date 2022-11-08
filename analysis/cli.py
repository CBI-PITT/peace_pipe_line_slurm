import argparse


def main_parser():
    """

    Example:
    process_brains

    Run 'process_brains --help' for all options.

    :return: None
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--priority-files',
        dest="priority_files",
        type=str,
        required=False,
        help="Path(s) to the input ims file(s) to be analyzed"
    )
    parser.add_argument(
        '--scan-folders',
        dest="scan_folders",
        type=str,
        required=False,
        help="Path(s) to folder(s) to be scanned for ims files"
    )
    parser.add_argument(
        '-o',
        '--out-folder',
        dest="out_folder",
        type=str,
        required=False,
        help="Path to the analysis output (either of priority files or of folders that are scanned for ims)"
    )
    parser.add_argument(
        '--operations',
        dest="jobs",
        type=str,
        required=False,
        help="List of operations to do. Options include:"
            "pre_process, register_brain, get_best_registration, detect_cells,"
            "classify_cells, save_cells_imaris, analyze_cells_imaris,"
            "analyze_cells_cellfinder, visualize_cells"
    )
    parser.add_argument(
        '--pre-processing',
        dest="pre_proc_methods",
        type=str,
        required=False,
        help="List of pre-processing methods to use"
    )
    parser.add_argument(
        '--classification-model',
        dest="model_to_use",
        type=str,
        required=False,
        help="Custom model used by cellfinder for cell classification using NN"
    )
    parser.add_argument(
        '--use-dask',
        dest="use_dask",
        action="store_true",
        required=False,
        help="Use parallel computing via dask for computationally intense parts of the pipeline"
    )


    args = parser.parse_args()
    in_file_path = args.in_file_path
    network = args.network
    out_file_path = args.out_file_path
    path_to_model = args.path_to_model

    run_n2n(in_file_path, network=network, out_file_path=out_file_path, path_to_model=path_to_model)