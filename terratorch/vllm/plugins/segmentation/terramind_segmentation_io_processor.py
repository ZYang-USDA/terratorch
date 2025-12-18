# Copyright contributors to the Terratorch project

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from collections.abc import Sequence
from typing import Any, Optional, Union

import rasterio
import torch
from vllm.config import VllmConfig
from vllm.entrypoints.openai.protocol import IOProcessorRequest, IOProcessorResponse
from vllm.inputs.data import PromptType
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors.interface import IOProcessor, IOProcessorInput, IOProcessorOutput

from terratorch.tasks.tiled_inference import generate_tiled_inference_output, prepare_tiled_inference_input
from terratorch.vllm.plugins import generate_datamodule
from terratorch.cli_tools import write_tiff

from .types import PluginConfig, RequestData, RequestOutput, TiledInferenceParameters

logger = logging.getLogger(__name__)

class TerramindSegmentationIOProcessor(IOProcessor):
    """vLLM IOProcessor for segmentation tasks

    This class instantiates an IO Processor plugin for vLLM for pre/post processing of GeoTiff images
    to be used with Segmentation tasks.
    This plugin accepts GeoTiff images in the format of a url, a base64 encoded string or a file path.
    Similarly, it can generate GeoTiff images is the form of a base64 encoded string or a file path.

    The plugin accepts and returns data in various formats and can be configured via the below environment variable:
        TERRATORCH_SEGMENTATION_IO_PROCESSOR_CONFIG
    This variable is to be set while starting the vLLM instance.
    The plugins configurable variables are:
    - output_path (String): Default path for storing output files when requesting output in 'path' mode. It is is ignored otherwise.
    The full schema of the plugin configuration can be found in vllm.plugins.segmentation.types.PluginConfig
    

    Once instantiated from the vLLM side, the plugin is automatically used when performing inference requests to the
    '/pooling' endpoint of a vLLM instance.
    """

    def __init__(self, vllm_config: VllmConfig):

        super().__init__(vllm_config)

        self.model_config = vllm_config.model_config.hf_config.to_dict()["pretrained_cfg"]

        if not "data" in self.model_config:
            raise ValueError("The model config does not contain the "
                             "Terratorch datamodule configuration")

        plugin_config_string = os.getenv("TERRATORCH_SEGMENTATION_IO_PROCESSOR_CONFIG", "{}")

        self.plugin_config = PluginConfig.model_validate_json(plugin_config_string)

        self.datamodule = generate_datamodule(self.model_config["data"])
        
        self.tiled_inference_parameters = self._init_tiled_inference_parameters_info() 
        self.batch_size = 1
        self.requests_cache: dict[str, dict[str, Any]] = {}

    def _init_tiled_inference_parameters_info(self) -> TiledInferenceParameters:
        if "tiled_inference_parameters" in self.model_config["model"]["init_args"]:
            tiled_inf_param_dict = self.model_config["model"]["init_args"]["tiled_inference_parameters"]
            if not all(["h_crop" in tiled_inf_param_dict, "w_crop" in tiled_inf_param_dict]):
                if "crop" in tiled_inf_param_dict:
                    tiled_inf_param_dict["h_crop"] = tiled_inf_param_dict["crop"]
                    tiled_inf_param_dict["w_crop"] = tiled_inf_param_dict["crop"]
                    del tiled_inf_param_dict["crop"]
                else:
                    raise ValueError(f"Expect 'crop' (or 'h_crop' and 'w_crop') in tiled_inference_parameters "
                                    f"but got {tiled_inf_param_dict}")
            if ("stride" in tiled_inf_param_dict):
                tiled_inf_param_dict["h_stride"] = tiled_inf_param_dict["stride"]
                tiled_inf_param_dict["w_stride"] = tiled_inf_param_dict["stride"]
                del tiled_inf_param_dict["stride"]
        else:
            tiled_inf_param_dict = {}
        
        print(f"tiled_inference_parameters: {tiled_inf_param_dict}")
        return TiledInferenceParameters(**tiled_inf_param_dict)

    def parse_request(self, request: Any) -> IOProcessorInput:
        if type(request) is dict:
            image_prompt = RequestData(**request)
            return image_prompt
        if isinstance(request, IOProcessorRequest):
            if not hasattr(request, "data"):
                raise ValueError(
                    "missing 'data' field in OpenAIBaseModel Request")

            request_data = request.data

            if type(request_data) is dict:
                return RequestData(**request_data)
            else:
                raise ValueError("Unable to parse the request data")

        raise ValueError("Unable to parse request")

    def output_to_response(
            self, plugin_output: IOProcessorOutput) -> IOProcessorResponse:
        return IOProcessorResponse(
            request_id=plugin_output.request_id,
            data=plugin_output,
        )

    def pre_process(
        self,
        prompt: IOProcessorInput,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Union[PromptType, Sequence[PromptType]]:
        # Just run the async function froma. synchronous context.
        # Since we are already in the vLLM server event loop we use that one.
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.pre_process_async(prompt, request_id, **kwargs))


    async def pre_process_async(
        self,
        prompt: IOProcessorInput,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Union[PromptType, Sequence[PromptType]]:

        image_data = dict(prompt)
        dataset_path = image_data["data"]
        import copy
        data_module_config = copy.deepcopy(self.model_config["data"])
        data_module_config["init_args"]["data_root"] = dataset_path
        datamodule = generate_datamodule(data_module_config)

        datamodule.batch_size = 1
        datamodule.setup("predict")

        data_loader = datamodule.predict_dataloader()
        data = list(data_loader)[0]

        input_data = datamodule.aug(data)["image"]
        try:
            prompt_data, tensor_reshape_fn, input_batch_size, h_img, w_img, _ = (
                prepare_tiled_inference_input(input_data,
                                            h_crop=self.tiled_inference_parameters.h_crop,
                                            w_crop=self.tiled_inference_parameters.h_crop,
                                            h_stride=self.tiled_inference_parameters.h_stride,
                                            w_stride=self.tiled_inference_parameters.w_stride,
                                            delta=self.tiled_inference_parameters.delta)
            )
        except Exception:
            import traceback
            traceback.print_exc()


        prompts = []
        for tile in prompt_data:
            reshaped_tile = tensor_reshape_fn(tile.input_data)
            # TODO: Check if there's a better way of getting the data in the correct data type ouf of the box.
            vllm_input = {mod: tensor.to(torch.float16) for mod, tensor in reshaped_tile.items()}
            prompt = {
                "prompt_token_ids": [1],
                "multi_modal_data": vllm_input
            }

            prompts.append(prompt)

        # if no request_id is passed this means that the plugin is used with vlLM
        # in offline sync mode. Therefore, we assume that one request at a time is being processed
        if not request_id:
            request_id = "offline"
        self.requests_cache[request_id] = {
            "out_data_format": image_data["out_data_format"],
            "dataset_path": dataset_path,
            "prompt_data": prompt_data,
            "h_img": h_img,
            "w_img": w_img,
            "input_batch_size": input_batch_size,
            "filename": data["filename"][0],
        }

        return prompts

    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: Optional[str] = None,
        **kwargs,
    ) -> IOProcessorOutput:

        if not request_id:
            request_id = "offline"

        if request_id and (request_id in self.requests_cache):
            request_info = self.requests_cache[request_id]
            del(self.requests_cache[request_id])

        model_outputs = [output.outputs.data.squeeze(0) for output in model_output]
        outputs = list(zip(request_info["prompt_data"], model_outputs, strict=True))
        output = generate_tiled_inference_output(
            outputs=outputs,
            input_batch_size=request_info["input_batch_size"],
            h_img=request_info["h_img"],
            w_img=request_info["w_img"],
            delta=self.tiled_inference_parameters.delta
        )

        prediction = output.squeeze(0).argmax(dim=0).numpy()

        # retrieve original image metadata
        input_image_path = Path(request_info["dataset_path"]) / "DEM" / (request_info["filename"] + "_DEM.tif")
        out_file_path = Path(self.plugin_config.output_path) / (request_info["filename"] + "_prediction.tif")
        with rasterio.open(input_image_path, "r") as src:
            metadata = src.meta

        write_tiff(prediction, out_file_path, metadata)

        return RequestOutput(data_format=request_info["out_data_format"],
                                  data=str(out_file_path.resolve()),
                                  request_id=request_id)
