# pylint: disable=no-member, arguments-differ
# pylint: disable=unsubscriptable-object
"""API Entities Module

This module provides a comprehensive set of Pydantic models that represent
multiple entities obtained from the API . 

Classes:
    - Dataset: Represents a structure for datasets present in the models.
    - Model: Represents a structure for models fetched from the API.
    - Project: Represents a structure for projects retrieved from the API.
"""

from abc import ABC, abstractmethod
from typing import Literal, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from dhl_sdk._input_processing import (
    CultivationHistoricalPreprocessor,
    CultivationPropagationPreprocessor,
    Preprocessor,
    SpectraData,
    SpectraPreprocessor,
    format_predictions,
)
from dhl_sdk._constants import (
    DATASETS_URL,
    MODELS_URL,
    PREDICT_URL,
    PROJECTS_URL,
    TEMPLATES_URL,
)
from dhl_sdk._utils import (
    PredictionRequestConfig,
    Predictions,
    PredictionResponse,
    get_id_list,
)
from dhl_sdk.crud import Client, CRUDClient, DataBaseClient, Result
from dhl_sdk.db_entities import Experiment, Variable
from dhl_sdk.exceptions import (
    InvalidInputsException,
    ModelPredictionException,
    PredictionRequestException,
)


# pylint: disable=not-an-iterable
class Dataset(BaseModel):
    """Model Dataset"""

    id: str = Field(alias="id")
    name: str = Field(alias="name")
    description: str = Field(alias="description")
    variables: list[Variable] = Field(alias="variables")
    experiments: list[Experiment] = Field(default=[], alias="experiments")
    _client: Client = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._client = data["client"]

    def get_data(self, client: DataBaseClient) -> list[dict]:
        """Get the data from the experiments

        Returns:
        --------
            list[dict]: List with the experiment data organized by:
                key: variable code
                value: dictionary with 2 keys:
                    - "values": list with the values
                    - "timestamps": list with the timestamps for each value
        """

        run_data = []
        for experiment in self.experiments:
            data = experiment.get_data(client=client)
            run_data.append(data)

        return run_data

    @model_validator(mode="before")
    @classmethod
    def _unpack_reference_entities(cls, data) -> dict:
        def unpack_entity(source: str, entity):
            unpacked = []

            for i, info in enumerate(data[source]):
                try:
                    entity_id = info["id"]
                except KeyError as err:
                    raise KeyError(
                        f"The {source} index {i} does not contain an id"
                    ) from err

                var = entity.requests(data["client"]).get(entity_id)
                unpacked.append(var)

            data[source] = unpacked

        unpack_entity("variables", Variable)
        unpack_entity("experiments", Experiment)

        return data

    @staticmethod
    def requests(client: Client) -> CRUDClient["Dataset"]:
        # pylint: disable=missing-function-docstring
        return CRUDClient["Dataset"](client, DATASETS_URL, Dataset)


class SpectraDataset(Dataset):
    """Pydantic Model Dataset for Spectra"""

    _client: Client = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._client = data["client"]

    def get_spectra_index(self) -> int:
        """Get the index of the spectra variable"""
        for index, variable in enumerate(self.variables):
            if variable.variant == "spectrum":
                return index
        raise ValueError("No spectrum variable found in dataset")

    def get_spectra_code(self) -> str:
        """Get variable code of spectra variable"""
        for variable in self.variables:
            if variable.variant == "spectrum":
                return variable.code
        raise ValueError("No spectrum variable found in dataset")

    @staticmethod
    def requests(client: Client) -> CRUDClient["SpectraDataset"]:
        # pylint: disable=missing-function-docstring
        return CRUDClient["SpectraDataset"](client, DATASETS_URL, SpectraDataset)


class Model(BaseModel, ABC):
    """Pydantic BaseModel for predictive models from the API"""

    id: str = Field(alias="id")
    name: str = Field(alias="name")
    status: str = Field(alias="status")
    project_id: str = Field(alias="projectId")
    dataset: Dataset = Field(alias="dataset")
    config: dict = Field(alias="config")
    _client: Client = PrivateAttr()

    @property
    def success(self) -> bool:
        """Get the success status of the model"""
        return self.status == "success"

    @staticmethod
    @abstractmethod
    def requests(client: Client) -> CRUDClient["Model"]:
        """Resquests abstract method for Model Types"""

    def get_predictions(self, preprocessor: Preprocessor) -> dict:
        """Get the predictions for the model using selected strategy"""

        if preprocessor.validate():
            json_data = preprocessor.format()
        else:
            raise InvalidInputsException(
                "The provided inputs failed the validation step"
            )

        predictions = []
        for prediction_data in json_data:
            try:
                response = self._client.post(PREDICT_URL, prediction_data)
                response.raise_for_status()

                # in case of an error in the response (not HTTP)
                if "error" in response.json():
                    raise PredictionRequestException(response.json()["error"])

            except Exception as ex:
                raise ex

            predictions.append(PredictionResponse(**response.json()))

        return format_predictions(predictions, model=self)

    @property
    def model_variables(self) -> dict:
        """List of the variables used in the model"""

        model_variables = []
        groups: dict = self.config["groups"]

        for variable in self.dataset.variables:
            variable_id = variable.id
            for group, variables in groups.items():
                if variable_id in variables:
                    model_variables.append(variable)
                    if group == "Flows":
                        feed_concentrations = [
                            ref.concentration_id
                            for ref in variable.variant_details.references
                            if ref.concentration_id
                        ]
                        if feed_concentrations:
                            groups.update({variable_id: feed_concentrations})
                    break

        return model_variables

    def get_model_variables_codes(self) -> list[str]:
        """Get the codes of the variables used in the model"""

        return [variable.code for variable in self.model_variables]

    @abstractmethod
    def predict(self, **kwargs) -> Predictions:
        """Prediction for Model"""


class PredictionConfig(BaseModel):
    """Configuration class for prediction method. This configuration parameters are
    passed to the model in DHL.

    Parameters:
    -----------

    model_confidence: float, optional
        This parameter determines the range within which the predictions of the model are
        expected to fall, with a specified level of certainty, i.e, setting it to 80%
        corresponds to capturing the range between the 10th and 90th percentiles
        of the model's output. Must be a value between 1 and 99, by default 80
    """

    model_config = ConfigDict(protected_namespaces=())

    model_confidence: float = Field(default=80.0, ge=1.0, le=99.0)


class SpectraModel(Model):
    """Pydantic Model for Spectra Prediction Model from the API"""

    dataset: SpectraDataset = Field(alias="dataset")
    _spectra_size: Optional[int] = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        self._client = data["client"]

    @model_validator(mode="before")
    @classmethod
    def _validate_model_data(cls, data) -> dict:
        data["dataset"] = SpectraDataset(**data["dataset"], client=data["client"])
        return data

    def predict(
        self,
        spectra: SpectraData,
        inputs: Optional[dict] = None,
    ) -> Predictions:
        """
        Predicts the output of a given model for a given set of spectra.

        Parameters:
        -----------
        spectra : list[list[float]] or np.ndarray
            A 2D array representing spectra for prediction, where:
                - The first dimension corresponds to the spectra index.
                - The second dimension contains wavelength index.
        inputs : dict, optional
            Additional inputs to be used for prediction. The keys must be the Codes of the
            input variables, and the values must be lists of the same length as the number
            of spectrum, by default None.

        Returns:
        --------
        Dictionary with predictions where:
            key: variable code
            value: list with predictions for each spectrum

        Example:
        --------
        >>> spectra = [[1.0, 2.0, 3.0, 3.0, 4.0, 3.0], [2.0, 3.0, 4.0, 4.0, 5.0, 4.0]]
        >>> inputs = {"input1": [42, 33]}
        >>> result = model.predict(spectra, inputs)
        >>> print(result)
        {'output1': [prediction1, prediction2], 'output2': [prediction1, prediction2]}

        """

        if not self.success:
            raise ModelPredictionException(
                f"{self.name} is not ready for prediction. The current status is {self.status}"
            )

        spectra_processing_strategy = SpectraPreprocessor(
            spectra=spectra, inputs=inputs, model=self
        )

        predictions = super().get_predictions(spectra_processing_strategy)

        spectra_code = self.dataset.get_spectra_code()
        if spectra_code in predictions:
            predictions.pop(spectra_code)

        return predictions

    @property
    def inputs(self) -> list[str]:
        """Get the inputs from the model's config"""
        return self.config["groups"]["Inputs"]

    @property
    def outputs(self) -> list[str]:
        """Get the outputs from the model's config"""
        return self.config["groups"]["Outputs"]

    @property
    def spectra_size(self) -> int:
        """Get the size of the spectra"""
        if self._spectra_size is None:
            self._spectra_size = self._get_spectra_size()
        return self._spectra_size

    def _get_spectra_size(self) -> int:
        """Get the size of the spectra from variable information in the API"""
        spectrum = self.dataset.variables[self.dataset.get_spectra_index()]
        return spectrum.size

    @staticmethod
    def requests(client: Client) -> CRUDClient["SpectraModel"]:
        # pylint: disable=missing-function-docstring
        return CRUDClient["SpectraModel"](client, MODELS_URL, SpectraModel)


class CultivationModel(Model, ABC):
    """Abstract Pydantic Model for Cultivation Prediction Model from the API"""

    @model_validator(mode="before")
    @classmethod
    def _validate_model_data(cls, data) -> dict:
        data["dataset"] = Dataset(**data["dataset"], client=data["client"])
        return data

    @abstractmethod
    def predict(
        self,
        timestamps: list,
        inputs: dict,
        timestamps_unit: str = "s",
        config: PredictionConfig = PredictionConfig(),
    ) -> dict:
        """Prediction for CultivationModel"""

    @staticmethod
    @abstractmethod
    def requests(client: Client) -> CRUDClient["CultivationModel"]:
        """CRUDClient for CultivationModel"""


class CultivationPropagationModel(CultivationModel):
    """Pydantic Model for Propagation Model for Cultivation from the API"""

    def __init__(self, **data):
        super().__init__(**data)
        self._client = data["client"]

    def predict(
        self,
        timestamps: list,
        inputs: dict,
        timestamps_unit: str = "s",
        config: PredictionConfig = PredictionConfig(),
    ) -> dict:
        """
        Predicts the output of a given model for a given set of inputs.

        Parameters:
        -----------
        timestamps : list
            A list of timestamps for prediction.
        inputs : dict, optional
            Inputs to be used for prediction. The keys must be the Codes of the
            input variables, and the values must be lists of the same length as the timestamps.
        timestamps_unit : str, optional
            Unit of the timestamps, by default "s".
            Needs to be one of the following: "s", "m", "h", "d".
        config: PredictionConfig, optional
            Configuration for the prediction method. Refer to the PredictionConfig class for
            details on the configuration parameters.
            See also: `PredictionConfig`


        Returns:
        --------
        Dictionary with predictions where:
            key: variable code
            value: list with predictions for each spectrum

        Example:
        --------
        >>> timestamps = [1,2,3,4,5,6,7]
        >>> inputs = {"var1": [42], "var2": [0.3], "var3": [0.5], "var4": [0,2,3,3,3,3,3]}
        >>> result = model.predict(timestamps, inputs, timestamps_unit="d")
        >>> print(result)
        {'var2': [pred1, pred2, pred3, pred4, pred5, pred6, pred7],
        'var3': [pred1, pred2, pre3, pred4, pred5, pred6, pred7]}

        """

        if not self.success:
            raise ModelPredictionException(
                f"{self.name} is not ready for prediction. The current status is {self.status}"
            )

        prediction_config = PredictionRequestConfig.new(
            model_confidence=config.model_confidence
        )

        data_processing_strategy = CultivationPropagationPreprocessor(
            timestamps=timestamps,
            timestamps_unit=timestamps_unit,
            inputs=inputs,
            prediction_config=prediction_config,
            model=self,
        )

        return super().get_predictions(data_processing_strategy)

    @staticmethod
    def requests(client: Client) -> CRUDClient["CultivationPropagationModel"]:
        # pylint: disable=missing-function-docstring
        return CRUDClient["CultivationPropagationModel"](
            client, MODELS_URL, CultivationPropagationModel
        )


class CultivationHistoricalModel(CultivationModel):
    """Pydantic Model for Historical Model for Cultivation from the API"""

    def __init__(self, **data):
        super().__init__(**data)
        self._client = data["client"]

    # pylint: disable=arguments-renamed
    def predict(
        self,
        timestamps: list[Union[int, float]],
        steps: list[Optional[int]],
        inputs: dict[str, list],
        timestamps_unit: str = "s",
        config: PredictionConfig = PredictionConfig(),
    ) -> dict:
        """
        Predicts the output of a given model for a given set of inputs.

        Parameters:
        -----------
        timestamps : list
            A list of timestamps for prediction.
        steps : list
            A list of steps for prediction. This steps should match the length of the timestamps
            and start as 0, representing the steps from the start of the process.
        inputs : dict, optional
            Inputs to be used for prediction. The keys must be the Codes of the
            input variables, and the values must be lists of the same length as the timestamps.
        timestamps_unit : str, optional
            Unit of the timestamps, by default "s".
            Needs to be one of the following: "s", "m", "h", "d".
        config: PredictionConfig, optional
            Configuration for the prediction method. Refer to the PredictionConfig class for
            details on the configuration parameters.
            See also: `PredictionConfig`

        Returns:
        --------
        Dictionary with predictions where:
            key: variable code
            value: list with predictions for each spectrum

        Example:
        --------
        >>> timestamps = [86400, 172800, 259200, 345600, 432000, 518400, 604800]
        >>> steps = [0,1,2,3,4,5,6]
        >>> inputs = {"var1": [42], "var2": [0.3], "var3": [0.5], "var4": [0,2,3,3,3,3,3]}
        >>> result = model.predict(timestamps, steps, inputs, timestamps_unit="s")
        >>> print(result)
        {'output1': [pred1], 'output2': [pred2]}

        """

        if not self.success:
            raise ModelPredictionException(
                f"{self.name} is not ready for prediction. The current status is {self.status}"
            )

        prediction_config = PredictionRequestConfig.new(
            model_confidence=config.model_confidence
        )

        data_processing_strategy = CultivationHistoricalPreprocessor(
            timestamps=timestamps,
            timestamps_unit=timestamps_unit,
            steps=steps,
            inputs=inputs,
            prediction_config=prediction_config,
            model=self,
        )

        return super().get_predictions(data_processing_strategy)

    @staticmethod
    def requests(client: Client) -> CRUDClient["CultivationHistoricalModel"]:
        # pylint: disable=missing-function-docstring
        return CRUDClient["CultivationHistoricalModel"](
            client, MODELS_URL, CultivationHistoricalModel
        )


# pylint: disable=too-few-public-methods
class ModelFactory:
    """Factory for Model, given the process unit id and model type"""

    MODEL_MAP = {
        "373c173a-1f23-4e56-874e-90ca4702ec0d": SpectraModel,
        "04a324da-13a5-470b-94a1-bda6ac87bb86": CultivationModel,
    }

    def __init__(self, process_unit_id):
        self._process_unit_id = process_unit_id

    def get_model(self, **kwargs) -> Type[Model]:
        """Get the model type from the process unit id"""

        if self._process_unit_id not in self.MODEL_MAP:
            raise NotImplementedError(
                f"Process unit id {self._process_unit_id} is not currently supported"
            )

        model = self.MODEL_MAP[self._process_unit_id]

        if "model_type" in kwargs and model == CultivationModel:
            if kwargs["model_type"] == "propagation":
                model = CultivationPropagationModel
            elif kwargs["model_type"] == "historical":
                model = CultivationHistoricalModel

        return model


class Project(BaseModel, ABC):
    """Abstract class for a DHL Project"""

    id: str = Field(alias="id")
    name: str = Field(alias="name")
    description: str = Field(alias="description")
    process_unit_id: str = Field(alias="processUnitId")
    _client: Client = PrivateAttr()

    def get_datasets(self, name: Optional[str] = None) -> Result[Dataset]:
        """Get the datasets of the project from the API

        Parameters:
        -----------
        name: str, optional
            Name of the dataset to be retrieved, by default None

        Returns:
        --------

        Result[Dataset]: Iteratable object containing the that correspond to the search

        """

        datasets = Dataset.requests(self._client)
        query_params = {
            "filterBy[projectId]": self.id,
        }

        if name is not None:
            query_params.update({"filterBy[name]": name})

        results = Result[Dataset](
            limit=5,
            query_params=query_params,
            requests=datasets,
        )

        return results

    @abstractmethod
    def get_models(self, name: Optional[str] = None, **kwargs) -> Result[Model]:
        """Get the models of the project from the API

        Parameters:
        -----------
        name: str, optional
            Name of the model to be retrieved, by default None
        model_type: str, optional
            For Cultivation Models, the type of the model to be retrieved,
            it can be `propagation` or `historical`, by default `propagation`
        """

    @abstractmethod
    def _get_model_query_params(self, name: Optional[str] = None) -> dict[str, str]:
        """Get the query params for the models"""

    @staticmethod
    @abstractmethod
    def requests(client: Client) -> CRUDClient["Project"]:
        """Resquests abstract method for Project Types"""


class SpectraProject(Project):
    """Pydantic Model for a DHL Project from the API"""

    def __init__(self, **data):
        super().__init__(**data)
        self._client = data["client"]

    def get_models(self, name: Optional[str] = None) -> Result[Model]:
        """Get the models of the project from the API"""

        model = ModelFactory(self.process_unit_id).get_model()

        models = model.requests(self._client)
        query_params = self._get_model_query_params(name=name)

        results = Result[model](
            limit=5,
            query_params=query_params,
            requests=models,
        )

        return results

    def _get_model_query_params(self, name: Optional[str] = None) -> dict[str, str]:
        query_params = {"filterBy[projectId]": self.id}

        if name is not None:
            query_params.update({"filterBy[name]": name})

        return query_params

    @staticmethod
    def requests(client: Client) -> CRUDClient["SpectraProject"]:
        # pylint: disable=missing-function-docstring
        return CRUDClient["SpectraProject"](client, PROJECTS_URL, SpectraProject)


class CultivationProject(Project):
    """Pydantic Model for a DHL Project from the API"""

    id: str = Field(alias="id")
    name: str = Field(alias="name")
    description: str = Field(alias="description")
    process_unit_id: str = Field(alias="processUnitId")
    _client: Client = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._client = data["client"]

    def get_models(
        self,
        name: Optional[str] = None,
        model_type: Literal["propagation", "historical"] = "propagation",
    ) -> Result[Model]:
        """Get the models of the project from the API"""

        if model_type not in ["propagation", "historical"]:
            raise ValueError(
                f"model_type must be either propagation or historical, got {model_type}"
            )

        query_params = self._get_model_query_params(name=name, model_type=model_type)

        model = ModelFactory(self.process_unit_id).get_model(model_type=model_type)
        models = model.requests(self._client)

        results = Result[model](
            limit=5,
            query_params=query_params,
            requests=models,
        )
        return results

    def _get_model_query_params(
        self,
        name: Optional[str] = None,
        model_type: Literal["propagation", "historical"] = "propagation",
    ) -> dict[str, str]:
        query_params = {"filterBy[projectId]": self.id}

        if name is not None:
            query_params.update({"filterBy[name]": name})

        # get templateIds for propagation models
        template_query_params = {
            "filterByTag[type]": model_type,
            "archived": "any",
        }
        template_list = self._client.get(TEMPLATES_URL, template_query_params).json()
        template_ids = get_id_list(template_list)

        query_params.update({"filterBy[templateId]": "|".join(template_ids)})

        return query_params

    @staticmethod
    def requests(client: Client) -> CRUDClient["CultivationProject"]:
        # pylint: disable=missing-function-docstring
        return CRUDClient["CultivationProject"](
            client, PROJECTS_URL, CultivationProject
        )
