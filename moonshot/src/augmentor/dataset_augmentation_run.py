from __future__ import annotations

import asyncio
import time

# from datetime import datetime
from typing import Callable

from slugify import slugify

from moonshot.src.configs.env_variables import EnvVariables
from moonshot.src.runners.runner_type import RunnerType
from moonshot.src.storage.storage import Storage
from moonshot.src.utils.import_modules import get_instance
from moonshot.src.utils.log import configure_logger

# Create a logger for this module
logger = configure_logger(__name__)


class DatasetAugmentationRun:
    def __init__(
        self,
        runner_id: str,
        runner_type: RunnerType,
        runner_args: dict,
        progress_callback_func: Callable | None = None,
    ):
        # created_epoch = time.time()
        # created_datetime = datetime.fromtimestamp(created_epoch).strftime(
        #     "%Y%m%d-%H%M%S"
        # )
        self.runner_id = slugify(runner_id, lowercase=True)
        self.runner_type = runner_type
        self.runner_args = runner_args
        self.progress_callback_func = progress_callback_func
        self.cancel_event = asyncio.Event()

    async def run(self) -> list | None:
        """
        Executes the dataset augmentation run.

        This method performs the following steps:
        1. Gets the asyncio running loop.
        2. Loads the runner processing module.
        3. Runs the runner processing module.
        4. Wraps up the run and returns the results.

        Returns:
            list | None: The results from the runner processing module, or None if an error occurs.
        """
        # ------------------------------------------------------------------------------
        # Part 1: Get asyncio running loop
        # ------------------------------------------------------------------------------

        logger.debug("[Dataset Augmentation] Part 1: Loading asyncio running loop...")
        loop = asyncio.get_running_loop()

        # # ------------------------------------------------------------------------------
        # # Part 2: Load runner processing module
        # # ------------------------------------------------------------------------------
        logger.debug(
            "[Dataset Augmentation] Part 2: Loading runner processing module..."
        )
        start_time = time.perf_counter()
        runner_module_instance = None
        try:
            runner_processing_module_name = self.runner_args.get(
                "runner_processing_module", None
            )
            if runner_processing_module_name:
                # Intialize the runner instance
                runner_module_instance = get_instance(
                    runner_processing_module_name,
                    Storage.get_filepath(
                        EnvVariables.RUNNERS_MODULES.name,
                        runner_processing_module_name,
                        "py",
                    ),
                )
                if runner_module_instance:
                    runner_module_instance = runner_module_instance()
                else:
                    raise RuntimeError(
                        f"Unable to get defined runner module instance - {runner_module_instance}"
                    )
            else:
                raise RuntimeError(
                    f"Failed to get runner processing module name: {runner_processing_module_name}"
                )

        except Exception as e:
            logger.error(
                f"[Dataset Augmentation] Failed to load runner processing module in Part 2 due to error: {str(e)}"
            )
            raise e

        finally:
            logger.debug(
                f"[Dataset Augmentation] Loading runner processing module took "
                f"{(time.perf_counter() - start_time):.4f}s"
            )

        # ------------------------------------------------------------------------------
        # Part 3: Run runner processing module
        # ------------------------------------------------------------------------------
        logger.debug("[Data Augmentation] Part 3: Running runner processing module...")
        start_time = time.perf_counter()
        runner_results = {}

        try:
            if runner_module_instance:
                runner_results = await runner_module_instance.generate(  # type: ignore ; ducktyping
                    loop,
                    self.runner_args,
                    self.cancel_event,
                )
            else:
                raise RuntimeError("Failed to initialise runner module instance.")

        except Exception as e:
            logger.error(
                f"[Data Augmentation] Failed to run runner processing module in Part 3 due to error: {str(e)}"
            )
            raise e

        finally:
            logger.debug(
                f"[Data Augmentation] Running runner processing module took {(time.perf_counter() - start_time):.4f}s"
            )

        # ------------------------------------------------------------------------------
        # Part 4: Wrap up run
        # ------------------------------------------------------------------------------
        logger.debug("[Data Augmentation] Part 4: Wrap up run...")
        return runner_results
