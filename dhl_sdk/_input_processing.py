"""This module contains utility functions for data validation and formatting in the SDK
"""

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=too-few-public-methods

import math

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol, Union

import numpy as np

from dhl_sdk._spectra_utils import (
    SpectraData,
    SpectraModel,
    _convert_to_request,
    _validate_spectra_format,
)
from dhl_sdk._utils import (
    Metadata,
    PipelineStage,
    PredictionPipelineRequest,
    PredictionRequestConfig,
    Predictions,
    PredictionResponse,
    validate_list_elements,
)
from dhl_sdk.exceptions import (
    InvalidInputsException,
    InvalidSpectraException,
    InvalidStepsException,
    InvalidTimestampsException,
)


class Group(Protocol):
    @property
    def code(self) -> str:
        ...


class Variable(Protocol):
    @property
    def group(self) -> Group:
        ...

    @property
    def code(self) -> str:
        ...

    @property
    def id(self) -> str:
        ...

    def matches_key(self, key: str) -> bool:
        ...


class Dataset(Protocol):
    @property
    def variables(self) -> list[Variable]:
        ...


class Model(Protocol):
    id: str

    @property
    def dataset(self) -> Dataset:
        ...

    @property
    def model_variables(self) -> list[Variable]:
        ...


# Input Preprocessors (Validation and Formatting)
class Preprocessor(ABC):
    """Interface for preprocessing"""

    @abstractmethod
    def validate(self) -> bool:
        """Validate the input data"""

    @abstractmethod
    def format(self) -> list[dict]:
        """Format the input data to a list of JSON requests"""


@dataclass
class SpectraPreprocessor(Preprocessor):
    """Implementation of the preprocessor for Spectrum models"""

    spectra: SpectraData
    inputs: Optional[dict]
    model: SpectraModel

    def validate(self) -> bool:
        """
        This function validates and makes the necessary formatting of the spectra
        and inputs used for prediction. It performs the following validations:

        - Validates if spectra are empty
        - Validates if the number of wavelengths in the spectra match the
            number of wavelengths in the model
        - Validates if the spectra contains None (missing) values
        - Validates if the number of inputs matches model inputs
        - Validates if the number of values in each input matches the number of spectra

        Parameters
        ----------
        spectra : Union[list[list[float]], np.ndarray]
            The spectra to be validated
        model : Model
            The model to use for prediction.
        inputs : dict, optional
            A dictionary of input variables and their values, by default None

        Returns
        -------
        (list[list[float]], dict)
            If all validations pass, return the formatted spectra and inputs
            as a tuple.

        Raises
        ------
        InvalidSpectraException
            Exception raised when Spectra are not valid for prediction.
        InvalidInputsException
            Exception raised when Inputs are not valid for prediction.
        """

        # Validate if empty spectra
        n_spectra = len(self.spectra)
        if n_spectra < 1:
            raise InvalidSpectraException("Empty spectra provided")

        self.spectra = _validate_spectra_format(self.spectra)

        # Validate number of wavelengths in spectra
        for i, spectrum in enumerate(self.spectra):
            if len(spectrum) != self.model.spectra_size:
                raise InvalidSpectraException(
                    (
                        f"Invalid Spectra: The Number of Wavelengths does not"
                        f"match training data for spectrum number: {i+1}. "
                        f"Expected: {self.model.spectra_size}, Got: {len(spectrum)}"
                    )
                )
            if validate_list_elements(spectrum):
                raise InvalidSpectraException(
                    (
                        f"Invalid Spectra: The Spectra contains not "
                        f"valid values for spectrum number: {i+1}"
                    )
                )

        if self.inputs is None:
            if len(self.model.inputs) > 0:
                raise InvalidInputsException(
                    "The model requires inputs, but none were provided."
                )
            return True

        if len(self.model.inputs) == 0:
            raise InvalidInputsException(
                "The model does not require inputs, but some were provided."
            )

        # validate inputs with spectra inputs (number of lines)
        for key, value in self.inputs.items():
            if len(value) != n_spectra:
                raise InvalidInputsException(
                    f"The Number of values does not match the number of spectra for input: {key}"
                )
            if validate_list_elements(value):
                raise InvalidInputsException(
                    f"Invalid Inputs: The Inputs contain non-valid values for input: {key}"
                )

        return True

    def format(self) -> list[dict]:
        """
        Format the inputs for a given model.

        Returns
        -------
        list[dict]
            A list of JSON requests, where each request contains a list of instances.

        Raises
        ------
        InvalidInputsException
            If no matching input is found for a given key.
        """

        if self.inputs is None:
            return _convert_to_request(self.spectra, model=self.model)

        model_variables = self.model.dataset.variables

        input_variables = [
            variable for variable in model_variables if variable.id in self.model.inputs
        ]

        # validate inputs codes and format for ids
        formatted_inputs = {}
        for key, value in self.inputs.items():
            for variable in input_variables:
                if variable.matches_key(key):
                    formatted_inputs[variable.id] = value
                    break
            else:
                correct_inputs = [print(variable) for variable in input_variables]
                raise InvalidInputsException(
                    (
                        f"No matching Input found for key: {key}. "
                        f"Please select one of the following as inputs: {*correct_inputs,}"
                    )
                )

        return _convert_to_request(
            self.spectra, model=self.model, inputs=formatted_inputs
        )


@dataclass
class CultivationPropagationPreprocessor(Preprocessor):
    """Implementation of the preprocessing strategy for cultivation models"""

    timestamps: Union[list[Union[int, float]], np.ndarray]
    timestamps_unit: str
    inputs: dict[str, list]
    prediction_config: PredictionRequestConfig
    model: Model

    def validate(self) -> bool:
        """
        This method validates the timestamps and inputs used for prediction.
        It performs the following validations:

        - Validates the structure and values of the timestamps
            - Valid timestamps should be a list of integers, have a length of at least 2,
            contain valid numeric values,be in ascending order, have positive values and be unique.
        - Validates the structure and values of the inputs
            - The inputs should be organized with variable codes as
                keys and lists of input values as values.
        - Validates the inputs based on the model variables
            - Garantees that all the mandatory model variables are present in the inputs
            - Garantees that the inputs that need to be complete for the recipe
                have the same length as the timestamps
            - Garantees that the initial condition variables only have one value

        Returns
        -------
        bool
            True if all validations pass

        Raises
        ------
        InvalidTimestampsException
            Exception raised when Timestamps are not valid for prediction.
        InvalidInputsException
            Exception raised when Inputs are not valid for prediction.
        """

        if isinstance(self.timestamps, np.ndarray):
            self.timestamps = self.timestamps.tolist()

        # Validate timestamps
        self.timestamps = _validate_upstream_timestamps(
            timestamps=self.timestamps, timestamps_unit=self.timestamps_unit
        )

        # Validate inputs
        _validate_upstream_inputs(inputs=self.inputs)

        # validate inputs with model variables
        _validate_propagation_with_variables(self.timestamps, self.inputs, self.model)
        return True

    def format(self) -> list[dict]:
        """
        This method formats the timestamps and inputs to be in the format required by the API.:

        Returns
        -------
        list[dict]
            list of dictionaries with instances for prediction
        """

        input_variables = self.model.dataset.variables
        model_variables = self.model.model_variables

        instances = [[]]

        formatted_inputs = {}

        # order the dict according to Variables and insert timestamps
        for key, value in self.inputs.copy().items():
            for variable in model_variables:
                if variable.matches_key(key):
                    formatted_inputs[variable.id] = {}
                    formatted_inputs[variable.id]["values"] = value
                    formatted_inputs[variable.id]["timestamps"] = self.timestamps[
                        : len(value)
                    ]
                    break

        for variable in input_variables:
            if variable.id in formatted_inputs:
                instances[0].append(formatted_inputs[variable.id])
            else:
                instances[0].append(None)

        json_data = PredictionPipelineRequest(
            instances=instances,
            metadata=Metadata(
                variables=[{"id": var.id} for var in input_variables],
            ),
            stages=[PipelineStage(config=self.prediction_config, id=self.model.id)],
        ).model_dump(
            by_alias=True,
            exclude_none=True,
            include={
                "instances": {"__all__": {"__all__": {"timestamps", "values"}}},
                "metadata": True,
                "stages": True,
            },
        )

        return [json_data]


@dataclass
class CultivationHistoricalPreprocessor(Preprocessor):
    """Implementation of the preprocessing strategy for cultivation models"""

    timestamps: Union[list[Union[int, float]], np.ndarray]
    timestamps_unit: str
    steps: list[Optional[int]]
    inputs: dict[str, list]
    prediction_config: PredictionRequestConfig
    model: Model

    def validate(self) -> bool:
        """
        This method validates the timestamps and inputs used for prediction.
        It performs the following validations:

        - Validates the structure and values of the timestamps
            - Valid timestamps should be a list of integers, have a length of at least 2,
            contain valid numeric values, in ascending order, have positive values and be unique.
        - Validates the structure and values of steps
            - Valid steps should be a list of integers, contain valid numeric values, in ascending \
                  order, have positive values and be unique.
        - Validates the structure and values of the inputs
            - The inputs should be organized with variable codes as \
                keys and lists of input values as values.

        Returns
        -------
        bool
            True if all validations pass

        Raises
        ------
        InvalidTimestampsException
            Exception raised when Spectra is not valid for prediction.
        InvalidInputsException
            Exception raised when Inputs is not valid for prediction.
        """

        if isinstance(self.timestamps, np.ndarray):
            self.timestamps = self.timestamps.tolist()

        # Validate timestamps
        self.timestamps = _validate_upstream_timestamps(
            timestamps=self.timestamps, timestamps_unit=self.timestamps_unit
        )

        # Validate steps
        _validate_historical_steps(steps=self.steps, timestamps=self.timestamps)

        # Validate inputs
        _validate_upstream_inputs(inputs=self.inputs)

        # # validate inputs with model variables
        _validate_historical_with_variables(self.inputs, self.model)

        return True

    def format(self) -> list[dict]:
        """
        This method formats the timestamps and inputs to be in the format required by the API.

        Returns
        -------
        list[dict]
            list of dictionaries with instances for prediction
        """

        input_variables = self.model.dataset.variables
        model_variables = self.model.model_variables

        instances = [[]]

        formatted_inputs = {}

        # order the dict according to Variables and insert timestamps and steps
        for key, value in self.inputs.copy().items():
            for variable in model_variables:
                if variable.matches_key(key):
                    formatted_inputs[variable.id] = {}
                    formatted_inputs[variable.id]["values"] = value
                    formatted_inputs[variable.id]["timestamps"] = self.timestamps[
                        : len(value)
                    ]
                    formatted_inputs[variable.id]["steps"] = self.steps[: len(value)]
                    break

        for variable in input_variables:
            if variable.id in formatted_inputs:
                instances[0].append(formatted_inputs[variable.id])
            else:
                instances[0].append(None)

        json_data = PredictionPipelineRequest(
            instances=instances,
            metadata=Metadata(
                variables=[{"id": var.id} for var in input_variables],
            ),
            stages=[PipelineStage(config=self.prediction_config, id=self.model.id)],
        ).model_dump(
            by_alias=True,
            exclude_none=True,
            include={
                "instances": {
                    "__all__": {"__all__": {"timestamps", "values", "steps"}}
                },
                "metadata": True,
                "stages": True,
            },
        )

        return [json_data]


def _validate_upstream_timestamps(
    timestamps: list[Union[int, float]], timestamps_unit: str
) -> list[int]:
    """Validate the timestamps for upstream prediction.

    This function performs a series of validations on the provided timestamps to ensure they meet
    the requirements for upstream prediction tasks.

    Parameters
    ----------
    timestamps : list[Union[int, float]]
        List of timestamps in the specified unit of time
    timestamps_unit : str
        Unit of the timestamps. Can be one of the following:
        - 's', 'sec', 'secs', 'seconds'
        - 'm', 'min', 'mins', 'minutes'
        - 'h', 'hour', 'hours'
        - 'd', 'day', 'days'

    Returns
    -------
    list[Union[int, float]]
        List of timestamps in seconds

    Raises
    ------
    InvalidTimestampsException
        If timestamps do not meet the specified criteria.

    """
    # Validate types
    if not isinstance(timestamps, list):
        raise InvalidTimestampsException("Timestamps must be a list of numbers")

    # Validate if length of timestamps
    if len(timestamps) <= 1:
        raise InvalidTimestampsException(
            "Timestamps must be a list of at least 2 values"
        )

    # Validate if timestamps are valid numeric values
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in timestamps
    ):
        raise InvalidTimestampsException(
            "All values of timestamps must be valid numeric values"
        )

    # Validate if timestamps are in ascending order and unique
    if any(a >= b for a, b in zip(timestamps, timestamps[1:])):
        raise InvalidTimestampsException("Timestamps must be in ascending order")

    # Validate if timestamps are positive (Since they are ordered, just check the first one)
    if timestamps[0] < 0:
        raise InvalidTimestampsException("Timestamps must be positive")

    # Convert timestamps to seconds
    unit_factors = {
        "s": 1,
        "sec": 1,
        "secs": 1,
        "seconds": 1,
        "m": 60,
        "min": 60,
        "mins": 60,
        "minutes": 60,
        "h": 3600,
        "hour": 3600,
        "hours": 3600,
        "d": 86400,
        "day": 86400,
        "days": 86400,
    }

    factor = unit_factors.get(timestamps_unit.lower())
    if factor is None:
        raise InvalidTimestampsException(
            f"Invalid timestamps unit '{timestamps_unit}' found."
        )

    # Apply the factor and convert to int
    return [int(timestamp * factor) for timestamp in timestamps]


def _validate_historical_steps(
    steps: list[Optional[int]], timestamps: list[int]
) -> None:
    """Validate the steps for historical prediction.

    This function performs a series of validations on the provided timestamps to ensure they meet
    the requirements for upstream prediction tasks.

    Parameters
    ----------
    steps : list[Optional[int]]
        List of steps, if None it means that a timestamp is not annotated with a step
    timestamps : list[int]
        List of timestamps corresponding to the steps

    Raises
    ------
    InvalidStepsException
        If steps do not meet the specified criteria.


    """
    # Validate types
    if not isinstance(steps, list):
        raise InvalidStepsException("Steps must be a list of numbers")

    # Validate if length of steps matches timestamps
    if len(steps) != len(timestamps):
        raise InvalidStepsException("Steps must have the same length as timestamps")

    # Validate if steps are in ascending order and unique
    filtered_steps = [step for step in steps if step is not None]

    if any(a >= b for a, b in zip(filtered_steps, filtered_steps[1:])):
        raise InvalidStepsException("Steps must be in ascending order")

    # Validate if steps start at 0
    if filtered_steps[0] != 0:
        raise InvalidStepsException("Steps must start at 0")


def _validate_upstream_inputs(inputs: dict[str, list]) -> None:
    """
    Validate the inputs for upstream prediction.
    The function checks if the inputs are a dictionary and if all values are lists.

    Parameters
    ----------
    inputs : dict[str, list]
        A dictionary, or list of dictionaries, where keys are variable codes,
        and values are lists of inputs.

    Raises
    ------
    InvalidInputsException
        If inputs do not meet the specified criteria.
    """

    if inputs is None:
        raise InvalidInputsException(
            "No Inputs provided. Please provide a dictionary of inputs"
        )

    # Validate types
    if not isinstance(inputs, dict):
        raise InvalidInputsException(
            "Inputs must be a dictionary of lists, with the variable code as key"
        )

    # Validate if inputs are lists
    if not all(isinstance(value, list) for value in inputs.values()):
        raise InvalidInputsException("All input values must be lists")


def _validate_propagation_with_variables(
    timestamps: list[int], inputs: dict[str, list], model: Model
) -> None:
    """
    Validate the inputs based on the provided timestamps and model.

    Parameters
    ----------
    timestamps : list[int]
        List of timestamps
    inputs : dict[str, list] | list[dict[str, list]]
        A dictionary where keys are variable codes, and values are lists of inputs.
    model : Model
        Model used for prediction

    Raises
    ------
    InvalidInputsException
        If inputs do not meet the specified criteria.
    """

    model_variables = model.model_variables

    for variable in model_variables:
        if variable.code in inputs:
            if groupcode_is_timedependent(variable.group.code):
                # Validate if inputs and timestamps are the same length
                if len(inputs[variable.code]) != len(timestamps):
                    raise InvalidInputsException(
                        (
                            f"The recipe requires {variable.code} to be complete,"
                            f"so it must have the same length as timestamps"
                        )
                    )

            else:
                # validate if non time dependent inputs have length of 1 (only initial values)
                if len(inputs[variable.code]) != 1:
                    raise InvalidInputsException(
                        (
                            f"Input {variable.code} only requires initial "
                            f"values, so it must have a length of 1"
                        )
                    )

            # Validate if numeric inputs are valid numeric values
            if groupcode_is_numeric(variable.group.code):
                if not all(
                    isinstance(value, (int, float)) and math.isfinite(value)
                    for value in inputs[variable.code]
                ):
                    raise InvalidInputsException(
                        f"All values of input {variable.code} must be valid numeric values"
                    )

        else:
            # validate if missing value is an Y Variable
            if not groupcode_is_output(variable.group.code):
                raise InvalidInputsException(
                    (
                        f"Input {variable.code} is a {variable.group.code} "
                        f"Variable, so it must be provided"
                    )
                )


def _validate_historical_with_variables(
    inputs: dict[str, list],
    model: Model,
) -> None:
    """
    Validate the inputs based on the provided timestamps and model.

    Parameters
    ----------
    timestamps : list[int]
        List of timestamps
    inputs : dict[str, list]
        A dictionary where keys are variable codes, and values are lists of inputs.
    model : Model
        Model used for prediction

    Raises
    ------
    InvalidInputsException
        If inputs do not meet the specified criteria.
    """

    model_variables = model.model_variables

    for variable in model_variables:
        if variable.code in inputs:
            if groupcode_is_numeric(variable.group.code):
                # Validate if inputs are valid numeric values
                if not all(
                    isinstance(value, (int, float)) for value in inputs[variable.code]
                ):
                    raise InvalidInputsException(
                        f"All values of input {variable.code} must be valid numeric values"
                    )

        else:
            # validate if missing value is an Y Variable
            if not groupcode_is_output(variable.group.code):
                raise InvalidInputsException(
                    (
                        f"Input {variable.code} is a {variable.group.code} "
                        f"Variable, so it must be provided"
                    )
                )


def format_predictions(
    predictions: list[PredictionResponse], model: Model
) -> Predictions:
    """Format a list of predictions into a dictionary.

    Parameters
    ----------
    predictions : List[PredictionResponse]
        list of predictions from the API.
    model : Model
        Model used for prediction

    Returns
    -------
    Dictionary with predictions where:
        key: variable id
        value: dictionary with keys "values" and, if applicable, "upperBound" and "lowerBound"

    """

    variables = [var.code for var in model.dataset.variables]

    dic = {}

    for pred in predictions:
        for i, instance in enumerate(pred.instances[0]):
            if instance is not None:
                if variables[i] not in dic:
                    dic[variables[i]] = {}

                    dic[variables[i]]["values"] = instance.values.copy()

                    if instance.high_values is not None:
                        dic[variables[i]]["upperBound"] = instance.high_values.copy()
                    if instance.low_values is not None:
                        dic[variables[i]]["lowerBound"] = instance.low_values.copy()
                else:
                    dic[variables[i]]["values"].extend(instance.values)

                    if instance.high_values is not None:
                        dic[variables[i]]["upperBound"].extend(instance.high_values)
                    if instance.low_values is not None:
                        dic[variables[i]]["lowerBound"].extend(instance.low_values)

    return dic


def variant_is_numeric(variant: str) -> bool:
    """Check if the variant is numeric"""
    return variant in ["flow", "numeric"]


def groupcode_is_numeric(code: str):
    """Check if the group is numeric"""
    return code in [
        "Flows",
        "FeedConc",
        "Inducers",
        "W",
        "X",
    ]


def groupcode_is_output(code: str):
    """Check if the group is an output / CQA"""
    return code == "Y"


def groupcode_is_timedependent(code: str):
    """Check if the group is time dependent for recipe only"""
    return code in ["Flows", "W", "Inducers"]
