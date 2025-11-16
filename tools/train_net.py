from __future__ import annotations

import logging

import torch
from lightning.pytorch import Trainer

from tools.common import build_training_components, default_argument_parser, load_config

LOGGER = logging.getLogger("tools.train_net")


def main() -> None:
    parser = default_argument_parser("Train SigLIP2-EoMT open-vocabulary panoptic model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    torch.set_float32_matmul_precision("high")

    cfg = load_config(args.config_file, args.opts)
    module, datamodule, trainer_kwargs, resume_path = build_training_components(
        cfg, resume=args.resume
    )

    trainer = Trainer(**trainer_kwargs)
    LOGGER.info("Starting training for %d epochs", trainer.max_epochs)
    trainer.fit(module, datamodule=datamodule, ckpt_path=resume_path)


if __name__ == "__main__":
    main()
