import time
import json

from dataclasses import dataclass


@dataclass
class LoggingArgs:
    """Arguments for logging attack results to txt, CSV, visdom, or wandb logs.
    
    Args:
        log_to_txt (str): Path to which to save attack logs as a text file. Set this argument if you want to save text logs.
            If the last part of the path ends with `.txt` extension, the path is assumed to path for output file.
        log_to_csv (str): Path to which to save attack logs as a CSV file. Set this argument if you want to save CSV logs.
            If the last part of the path ends with `.csv` extension, the path is assumed to path for output file.
        csv_coloring_style (str): Method for choosing how to mark perturbed parts of the text. Options are "file" and "plain".
            "file" wraps text with double brackets `[[ <text> ]]` while "plain" does not mark any text. Default is "file".
        log_to_visdom (dict): Set this argument if you want to log attacks to Visdom. The dictionary should have the following
            three keys and their corresponding values: `"env", "port", "hostname"` (e.g. `{"env": "main", "port": 8097, "hostname": "localhost"}`).
        log_to_wandb (str): Name of the wandb project. Set this argument if you want to log attacks to Wandb.
        disable_stdout (bool): Disable logging to stdout.
        
    """

    log_to_txt: str = None
    log_to_csv: str = None
    csv_coloring_style: str = "file"
    log_to_visdom: dict = None
    log_to_wand: str = None
    disable_stdout: str = False


    @classmethod
    def add_parser_args(cls, parser):
        """Adds dataset-related arguments to an argparser.

        This is useful because we want to load pretrained models using
        multiple different parsers that share these, but not all,
        arguments.
        """
        parser.add_argument(
            "--log-to-txt",
            nargs="?",
            default=None,
            const=""
            type=str,
            help="Path to which to save attack logs as a text file. Set this argument if you want to save text logs. "
                "If the last part of the path ends with `.txt` extension, the path is assumed to path for output file."
        )

        parser.add_argument(
            "--log-to-csv",
            nargs="?",
            default=None,
            const=""
            type=str,
            help="Path to which to save attack logs as a CSV file. Set this argument if you want to save CSV logs. "
                "If the last part of the path ends with `.csv` extension, the path is assumed to path for output file."
        )

        parser.add_argument(
            "--csv-coloring-style",
            default="file",
            type=str,
            help='Method for choosing how to mark perturbed parts of the text in CSV logs. Options are "file" and "plain". '
                '"file" wraps text with double brackets `[[ <text> ]]` while "plain" does not mark any text. Default is "file".'
        )

        parser.add_argument(
            "--log-to-visdom",
            nargs="?",
            default=None,
            const="{\"env\": \"main\", \"port\": 8097, \"hostname\": \"localhost\"}"
            type=json.loads,
            help='Set this argument if you want to log attacks to Visdom. The dictionary should have the following '
                'three keys and their corresponding values: `"env", "port", "hostname"`. '
                'Example for command line use: `--log-to-visdom {\"env\": \"main\", \"port\": 8097, \"hostname\": \"localhost\"}`.'
        )

        parser.add_argument(
            "--log-to-wandb",
            nargs="?",
            default=None,
            const="textattack"
            type=str,
            help="Name of the wandb project. Set this argument if you want to log attacks to Wandb."
        )

        parser.add_argument(
            "--disable-stdout", action="store_true", help="Disable logging to stdout"
        )

        return parser


    @classmethod
    def create_loggers_from_args(cls, args):
        assert isinstance(
            args, cls
        ), f"Expect args to be of type `{type(cls)}`, but got type `{type(args)}`."

        # Create logger
        attack_log_manager = textattack.loggers.AttackLogManager()

        # Get current time for file naming
        timestamp = time.strftime("%Y-%m-%d-%H-%M")

        # if '--log-to-txt' specified with arguments
        if args.log_to_txt is not None:
            if args.log_to_txt.lower().endswith(".txt"):
                txt_file_path = args.log_to_txt
            else:
                txt_file_path = os.path.join(args.log_to_txt, f"{timestamp}-log.txt")

            if not os.path.exists(os.path.dirname(txt_file_path)):
                os.makedirs(os.path.dirname(txt_file_path))

            attack_log_manager.add_output_file(txt_file_path)

        # if '--log-to-csv' specified with arguments
        if args.log_to_csv is not None:
            if args.log_to_csv.lower().endswith(".csv"):
                csv_file_path = args.log_to_csv
            else:
                csv_file_path = os.path.join(args.log_to_csv, f"{timestamp}-log.csv")

            if not os.path.exists(os.path.dirname(csv_file_path)):
                os.makedirs(os.path.dirname(csv_file_path))

            color_method = None if args.csv_coloring_style == "plain" else args.csv_coloring_style
            attack_log_manager.add_output_csv(csv_file_path, color_method)

        # Visdom
        if args.log_to_visdom is not None:
            attack_log_manager.enable_visdom(**args.log_to_visdom)

        # Weights & Biases
        if args.log_to_visdom is not None:
            attack_log_manager.enable_wandb(args.log_to_visdom)

        # Stdout
        if not args.disable_stdout:
            attack_log_manager.enable_stdout()

        return attack_log_manager
