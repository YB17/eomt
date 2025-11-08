from __future__ import annotations

import logging

from lightning.pytorch import Trainer

from tools.common import build_eval_components, default_argument_parser, load_config

LOGGER = logging.getLogger("tools.test_net")


def main() -> None:
    parser = default_argument_parser("Evaluate SigLIP2-EoMT open-vocabulary panoptic model")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path for evaluation")
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="val",
        choices=("val", "test"),
        help="Run validation or test loop",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    cfg = load_config(args.config_file, args.opts)
    module, datamodule, trainer_kwargs, _ = build_eval_components(cfg)

    trainer = Trainer(**trainer_kwargs)
    LOGGER.info("Running %s evaluation from %s", args.eval_mode, args.ckpt)
    if args.eval_mode == "test":
        trainer.test(module, datamodule=datamodule, ckpt_path=args.ckpt)
    else:
        trainer.validate(module, datamodule=datamodule, ckpt_path=args.ckpt)


if __name__ == "__main__":
    main()
