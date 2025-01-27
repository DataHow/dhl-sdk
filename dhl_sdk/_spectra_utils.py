"""This module contains utility functions used for spectra 
validation and formatting in the SDK
"""

from typing import Optional, Protocol, Union

import numpy as np

from dhl_sdk._utils import (
    Instance,
    Metadata,
    PipelineStage,
    PredictionPipelineRequest,
    SpectraPredictionConfig,
)
from dhl_sdk.exceptions import InvalidSpectraException

# Type Aliases
SpectraData = Union[
    list[list[float]],
    np.ndarray[tuple[int, int], Union[np.dtype[np.float32], np.dtype[np.float64]]],
]


class Dataset(Protocol):
    # pylint: disable=missing-class-docstring
    # pylint: disable=missing-function-docstring
    @property
    def variables(self) -> list:
        ...

    def get_spectra_index(self) -> int:
        ...


class SpectraModel(Protocol):
    # pylint: disable=missing-class-docstring
    # pylint: disable=missing-function-docstring
    @property
    def inputs(self) -> list[str]:
        ...

    @property
    def dataset(self) -> Dataset:
        ...

    @property
    def spectra_size(self) -> int:
        ...


def _validate_spectra_format(spectra: SpectraData) -> list[list[float]]:
    """
    Validates and formats the spectra.

    Parameters
    ----------
    spectra : list or numpy.ndarray
        The spectra to be validated.

    Returns
    -------
    list
        The spectra as a list.

    Raises
    ------
    InvalidSpectraException
        If the spectra is not a list or numpy array.
    """

    if isinstance(spectra, np.ndarray):
        spectra = spectra.tolist()

    elif isinstance(spectra, list):
        pass
    else:
        raise InvalidSpectraException(
            f"Spectra must be a list or numpy array, but got {type(spectra)}"
        )

    return spectra


def _convert_to_request(
    spectra: SpectraData,
    model: SpectraModel,
    inputs: Optional[dict] = None,
    batch_size: int = 50,
) -> list[dict]:
    """
    Convert spectra and inputs to a list of JSON requests.
    Handles big requests by paginating them.

    Parameters
    ----------
    spectra : list[list[float]]
        A list of spectra, where each spectrum is a list of floats.
    model : Model
        The model to use for prediction.
    inputs : dict, optional
        A dictionary of input variables and their values, by default None.
    batch_size : int, optional
        The maximum number of spectra to include in each request, by default 50.

    Returns
    -------
    list[dict]
        A list of JSON requests, where each request contains a list of instances.
    """

    # get number of vars in model from config
    variables = model.dataset.variables
    n_vars = len(variables)
    spectrum_index = model.dataset.get_spectra_index()

    request_data = []
    # handle pagination
    for i in range(0, len(spectra), batch_size):
        instance = [None] * n_vars
        instance[spectrum_index] = Instance(values=spectra[i : i + batch_size])

        if inputs is not None:
            for input_id, input_values in inputs.items():
                for index, variable in enumerate(variables):
                    if variable.id == input_id:
                        instance[index] = Instance(
                            values=input_values[i : i + batch_size]
                        )
                        break

        json_data = PredictionPipelineRequest(
            instances=[instance],
            metadata=Metadata(
                variables=[{"id": var.id} for var in model.dataset.variables],
            ),
            stages=[PipelineStage(config=SpectraPredictionConfig(), id=model.id)],
        ).model_dump(
            by_alias=True,
            exclude_none=True,
            include={
                "instances": True,
                "metadata": True,
                "stages": True,
            },
        )

        request_data.append(json_data)

    return request_data
