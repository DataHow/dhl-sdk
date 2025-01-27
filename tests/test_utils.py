# pylint: disable=missing-docstring
import unittest
from unittest.mock import Mock

import numpy as np
from pydantic import BaseModel

from dhl_sdk._input_processing import (
    CultivationHistoricalPreprocessor,
    CultivationPropagationPreprocessor,
    SpectraPreprocessor,
    _validate_spectra_format,
    format_predictions,
)
from dhl_sdk._utils import Instance, PredictionResponse, PredictionRequestConfig
from dhl_sdk.crud import Result
from dhl_sdk.entities import Variable
from dhl_sdk.exceptions import (
    InvalidInputsException,
    InvalidSpectraException,
    InvalidStepsException,
    InvalidTimestampsException,
)


class TestSpectraUtils(unittest.TestCase):
    def setUp(self):
        spectrum_var = {
            "id": "ram-111",
            "code": "spc1",
            "variant": "spectrum",
            "name": "raman 1",
            "spectrum": {"xAxis": {"dimension": 4}},
        }
        self.model_no_inputs = Mock()
        self.model_no_inputs.id = "model-id-1"
        self.model_no_inputs.inputs = []
        self.model_no_inputs.dataset.variables = [Variable(**spectrum_var)]
        self.model_with_inputs = Mock()
        self.model_with_inputs.id = "model-id-2"
        self.model_with_inputs.dataset.variables = [
            Variable(**spectrum_var),
            Variable(id="id-123", code="var1", variant="numeric", name="variable 1"),
            Variable(id="id-456", code="var2", variant="numeric", name="variable 2"),
            Variable(id="id-789", code="out1", variant="numeric", name="output 1"),
            Variable(id="id-101", code="out2", variant="numeric", name="output 2"),
        ]
        self.model_with_inputs.inputs = ["id-123", "id-456"]

    def test_format_spectra_validation(self):
        spectra1 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        spectra2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        spectra3 = "spectra"

        self.assertEqual(_validate_spectra_format(spectra1), spectra1)
        self.assertEqual(_validate_spectra_format(spectra2), spectra1)
        self.assertRaises(InvalidSpectraException, _validate_spectra_format, spectra3)

    def test_format_predictions(self):
        predictions = [
            PredictionResponse(
                instances=[
                    [
                        None,
                        None,
                        None,
                        Instance(
                            values=[1, 2, 3], highValues=[2, 3, 5], lowValues=[0, 1, 1]
                        ),
                        Instance(
                            values=[4, 5, 6], highValues=[5, 6, 6], lowValues=[1, 1, 1]
                        ),
                    ]
                ]
            ),
            PredictionResponse(
                instances=[
                    [
                        None,
                        None,
                        None,
                        Instance(
                            values=[1, 2, 3], highValues=[2, 3, 5], lowValues=[0, 1, 1]
                        ),
                        Instance(
                            values=[4, 5, 6], highValues=[5, 6, 6], lowValues=[1, 1, 1]
                        ),
                    ]
                ]
            ),
        ]

        formatted_predictions = format_predictions(
            predictions, model=self.model_with_inputs
        )

        self.assertDictEqual(
            formatted_predictions,
            {
                "out1": {
                    "values": [1, 2, 3, 1, 2, 3],
                    "upperBound": [2, 3, 5, 2, 3, 5],
                    "lowerBound": [0, 1, 1, 0, 1, 1],
                },
                "out2": {
                    "values": [4, 5, 6, 4, 5, 6],
                    "upperBound": [5, 6, 6, 5, 6, 6],
                    "lowerBound": [1, 1, 1, 1, 1, 1],
                },
            },
        )

    def test_validation_no_input(self):
        model = self.model_no_inputs
        model.spectra_size = 4

        spectra = [[1.0, 2.0, 3.0, 3.0], [4.0, 5.0, 6.0, 6.0], [7.0, 8.0, 9.0, 9.0]]
        processor = SpectraPreprocessor(spectra=spectra, model=model, inputs=None)
        processor.validate()

        empty_spectra = []
        processor = SpectraPreprocessor(spectra=empty_spectra, model=model, inputs=None)

        self.assertRaises(InvalidSpectraException, processor.validate)

        spectra = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        processor = SpectraPreprocessor(spectra=spectra, model=model, inputs=None)

        self.assertRaises(
            InvalidSpectraException,
            processor.validate,
        )

        spectra = [["1", "2", "3", "3"], [4, 5, 6, 6], [7, 8, 9, 9]]
        processor = SpectraPreprocessor(spectra=spectra, model=model, inputs=None)

        self.assertRaises(
            InvalidSpectraException,
            processor.validate,
        )

        spectra = [[1, 2, 3, 3], [4, 5, np.nan, 6], [7, 8, 9, 9]]
        processor = SpectraPreprocessor(spectra=spectra, model=model, inputs=None)

        self.assertRaises(
            InvalidSpectraException,
            processor.validate,
        )

        spectra = [[1.0, 2.0, 3.0, 3.0], [4.0, 5.0, 6.0, 6.0], [7.0, 8.0, 9.0]]
        processor = SpectraPreprocessor(spectra=spectra, model=model, inputs=None)

        self.assertRaises(
            InvalidSpectraException,
            processor.validate,
        )

        spectra = [[1.0, 2.0, 3.0, 3.0], [4.0, 5.0, 6.0, 6.0], [7.0, 8.0, 9.0, 9.0]]
        inputs = {"var1": [0, 1, 0], "var2": [1, 1, 1]}
        processor = SpectraPreprocessor(spectra=spectra, model=model, inputs=inputs)
        self.assertRaises(
            InvalidInputsException,
            processor.validate,
        )

    def test_validation_with_input(self):
        model = self.model_with_inputs
        model.spectra_size = 4

        spectra = [[1.0, 2.0, 3.0, 3.0], [4.0, 5.0, 6.0, 6.0], [7.0, 8.0, 9.0, 9.0]]
        processor = SpectraPreprocessor(spectra=spectra, inputs=None, model=model)

        self.assertRaises(
            InvalidInputsException,
            processor.validate,
        )

        inputs = {"var10": [0, 1, 0], "var2": [1, 1, 1]}
        processor = SpectraPreprocessor(spectra=spectra, model=model, inputs=inputs)

        with self.assertRaises(InvalidInputsException) as ex:
            processor.validate()
            processor.format()
            self.assertTrue(
                ex.exception.message.startswith(
                    "No matching Input found for key: var10"
                )
            )

        inputs = {"var1": [0, 1, 0], "var2": [1, 1, 1]}
        processor = SpectraPreprocessor(spectra=spectra, model=model, inputs=inputs)

        processor.validate()

        inputs = {"var1": [0, 1, 0], "var2": [1, 1, 1, 1]}
        processor = SpectraPreprocessor(spectra=spectra, model=model, inputs=inputs)
        with self.assertRaises(InvalidInputsException) as ex:
            processor.validate()
            self.assertTrue(
                ex.exception.message.startswith(
                    "The Number of values does not match the number of spectra"
                )
            )

        inputs = {"var1": [0, None, 0], "var2": [1, 1, 1]}
        processor = SpectraPreprocessor(spectra=spectra, model=model, inputs=inputs)
        with self.assertRaises(InvalidInputsException) as ex:
            processor.validate()
            self.assertTrue(
                ex.exception.message.startswith(
                    "Invalid Inputs: The Inputs contains not valid values for input: var1"
                )
            )

    def test_convert_to_request(self):
        model = self.model_with_inputs
        model.spectra_size = 4
        model.dataset.get_spectra_index.return_value = 0

        spectra = [[1.0, 2.0, 3.0, 3.0], [4.0, 5.0, 6.0, 6.0], [7.0, 8.0, 9.0, 9.0]]
        inputs = {"var1": [0, 1, 0], "var2": [1, 1, 1]}
        processor = SpectraPreprocessor(spectra=spectra, model=model, inputs=inputs)

        processor.validate()
        request = processor.format()

        self.assertEqual(request[0]["instances"][0][0]["sampleId"], ["0", "1", "2"])
        self.assertEqual(request[0]["instances"][0][0]["values"][0], [1, 2, 3, 3])
        self.assertEqual(request[0]["instances"][0][0]["values"][1], [4, 5, 6, 6])
        self.assertEqual(request[0]["instances"][0][0]["values"][2], [7, 8, 9, 9])
        self.assertEqual(request[0]["instances"][0][2]["values"], [1, 1, 1])

    def test_convert_request_noinput(self):
        model = self.model_no_inputs
        model.spectra_size = 4
        model.dataset.get_spectra_index.return_value = 0

        spectra = [[1.0, 2.0, 3.0, 3.0], [4.0, 5.0, 6.0, 6.0], [7.0, 8.0, 9.0, 9.0]]
        processor = SpectraPreprocessor(spectra=spectra, model=model, inputs=None)

        processor.validate()
        request = processor.format()

        self.assertEqual(request[0]["instances"][0][0]["sampleId"], ["0", "1", "2"])
        self.assertEqual(request[0]["instances"][0][0]["values"][0], [1, 2, 3, 3])
        self.assertEqual(request[0]["instances"][0][0]["values"][1], [4, 5, 6, 6])
        self.assertEqual(request[0]["instances"][0][0]["values"][2], [7, 8, 9, 9])


class TestCultivationUtils(unittest.TestCase):
    def setUp(self):
        var1 = {
            "id": "id-123",
            "code": "var1",
            "variant": "numeric",
            "name": "variable 1",
            "group": {"code": "X"},
        }
        var2 = {
            "id": "id-456",
            "code": "var2",
            "variant": "numeric",
            "name": "variable 2",
            "group": {"code": "X"},
        }
        var3 = {
            "id": "id-789",
            "code": "var3",
            "variant": "numeric",
            "name": "output 1",
            "group": {"code": "W"},
        }
        var4 = {
            "id": "id-101",
            "code": "var4",
            "variant": "numeric",
            "name": "output 2",
            "group": {"code": "Y"},
        }
        var5 = {
            "id": "id-999",
            "code": "var5",
            "variant": "numeric",
            "name": "not input",
            "group": {"code": "X"},
        }
        self.model = Mock()
        self.model.id = "model-id-1"
        self.model.dataset.variables = [
            Variable(**var1),
            Variable(**var2),
            Variable(**var3),
            Variable(**var4),
            Variable(**var5),
        ]

        self.model.model_variables = [
            Variable(**var1),
            Variable(**var2),
            Variable(**var3),
            Variable(**var4),
        ]

        self.prediction_config = PredictionRequestConfig()

    def test_format_predictions(self):
        predictions = [
            PredictionResponse(
                instances=[
                    [
                        Instance(values=[1, 2, 3]),
                        Instance(values=[4, 5, 6]),
                        None,
                        None,
                    ]
                ]
            ),
            PredictionResponse(
                instances=[
                    [
                        Instance(values=[1, 2, 3]),
                        Instance(values=[4, 5, 6]),
                        None,
                        None,
                    ]
                ]
            ),
        ]

        formatted_predictions = format_predictions(predictions, model=self.model)

        self.assertDictEqual(
            formatted_predictions,
            {
                "var1": {"values": [1, 2, 3, 1, 2, 3]},
                "var2": {"values": [4, 5, 6, 4, 5, 6]},
            },
        )

    def test_timestamp_validation(self):
        model = self.model
        inputs = {"var1": [10], "var2": [20], "var3": [1, 2, 3], "var4": [40]}
        timestamps = [1, 2, 3]

        processor = CultivationPropagationPreprocessor(
            np.array(timestamps), "m", inputs, self.prediction_config, model
        )
        processor.validate()
        processor.timestamps = [60, 120, 180]

        processor = CultivationPropagationPreprocessor(
            {"timestamps": [1, 2]}, "s", inputs, self.prediction_config, model
        )
        with self.assertRaises(InvalidTimestampsException) as ex:
            processor.validate()
            self.assertTrue(
                ex.exception.message.startswith("Timestamps must be a list of numbers")
            )

        processor = CultivationPropagationPreprocessor(
            [2], "s", inputs, self.prediction_config, model
        )
        with self.assertRaises(InvalidTimestampsException) as ex:
            processor.validate()
            self.assertTrue(
                ex.exception.message.startswith(
                    "Timestamps must be a list of at least 2 values"
                )
            )

        processor = CultivationPropagationPreprocessor(
            [1, 2], "s", inputs, self.prediction_config, model
        )
        with self.assertRaises(InvalidInputsException) as ex:
            processor.validate()
            self.assertTrue(
                ex.exception.message.startswith(
                    "The recipe requires var3 to be complete"
                )
            )

        processor = CultivationPropagationPreprocessor(
            [6, 4, 3], "s", inputs, self.prediction_config, model
        )

        with self.assertRaises(InvalidTimestampsException) as ex:
            processor.validate()
            self.assertTrue(
                ex.exception.message.startswith("Timestamps must be in ascending order")
            )

        processor = CultivationPropagationPreprocessor(
            [1, 2, 3], "m", inputs, self.prediction_config, model
        )
        processor.validate()
        self.assertEqual(processor.timestamps, [60, 120, 180])

        processor = CultivationPropagationPreprocessor(
            ["1", 2, 3], "h", inputs, self.prediction_config, model
        )

        with self.assertRaises(InvalidTimestampsException) as ex:
            processor.validate()
            self.assertTrue(
                ex.exception.message.startswith(
                    "All values of timestamps must be valid numeric values"
                )
            )

        processor = CultivationPropagationPreprocessor(
            [1, 2, 3], "random", inputs, self.prediction_config, model
        )
        with self.assertRaises(InvalidTimestampsException) as ex:
            processor.validate()
            self.assertTrue(
                ex.exception.message.startswith(
                    "Invalid timestamps unit 'random' found."
                )
            )

        processor = CultivationPropagationPreprocessor(
            [-1, 2, 4], "h", inputs, self.prediction_config, model
        )
        with self.assertRaises(InvalidTimestampsException) as ex:
            processor.validate()
            self.assertTrue(
                ex.exception.message.startswith("Timestamps must be positive")
            )

        processor = CultivationPropagationPreprocessor(
            [1, 2, 2], "h", inputs, self.prediction_config, model
        )
        with self.assertRaises(InvalidTimestampsException) as ex:
            processor.validate()
            self.assertTrue(
                ex.exception.message.startswith("Timestamps must be unique")
            )

    def test_input_validation(self):
        model = self.model
        inputs = {"var1": [10], "var2": [20], "var3": [1, 2, 3], "var4": [40]}
        timestamps = [1, 2, 3]

        processor = CultivationPropagationPreprocessor(
            timestamps, "s", inputs, self.prediction_config, model
        )

        processor.validate()

        inputs = {"var1": [10], "var3": [1, 2, 3], "var4": [40]}
        processor = CultivationPropagationPreprocessor(
            timestamps, "s", inputs, self.prediction_config, model
        )

        with self.assertRaises(InvalidInputsException) as ex:
            processor.validate()
            self.assertEqual(
                ex.exception.message,
                "Input var2 is a X Variable, so it must be provided",
            )

        inputs = {"var1": [10], "var2": [20], "var3": [1], "var4": [40]}
        processor = CultivationPropagationPreprocessor(
            timestamps, "s", inputs, self.prediction_config, model
        )

        with self.assertRaises(InvalidInputsException) as ex:
            processor.validate()
            self.assertTrue(
                ex.exception.message.startswith(
                    "The recipe requires var3 to be complete"
                )
            )

        inputs = {"var1": [10], "var2": [20], "var3": [1, 3, 5]}
        processor = CultivationPropagationPreprocessor(
            timestamps, "s", inputs, self.prediction_config, model
        )

        processor.validate()

        inputs = {"var1": [10, 20], "var2": [20], "var3": [1, 3, 5]}
        processor = CultivationPropagationPreprocessor(
            timestamps, "s", inputs, self.prediction_config, model
        )

        with self.assertRaises(InvalidInputsException) as ex:
            processor.validate()
            self.assertTrue(
                ex.exception.message.startswith(
                    "Input var1 only requires initial values"
                )
            )

        inputs = {"var1": [10], "var2": [20], "var3": [1, "a", 5], "var4": [40, 50]}
        processor = CultivationPropagationPreprocessor(
            timestamps, "s", inputs, self.prediction_config, model
        )

        with self.assertRaises(InvalidInputsException) as ex:
            processor.validate()
            self.assertTrue(
                ex.exception.message.startswith(
                    "All values of input var3 must be valid numeric values"
                )
            )

        inputs = {"var1": ["a"], "var2": [20], "var3": [1, 3, 5], "var4": [40, 50]}
        processor = CultivationPropagationPreprocessor(
            timestamps, "s", inputs, self.prediction_config, model
        )

        with self.assertRaises(InvalidInputsException) as ex:
            processor.validate()
            self.assertTrue(
                ex.exception.message.startswith(
                    "All values of input var1 must be valid numeric values"
                )
            )

    def test_historical_model_inputs(self):
        model = self.model
        inputs = {
            "var1": [10, 10, 10],
            "var2": [20, 20, 20],
            "var3": [1, 2, 3],
            "var4": [40],
        }
        timestamps = [1, 2, 3]
        steps = [0, 1, 2]

        processor = CultivationHistoricalPreprocessor(
            np.array(timestamps), "d", steps, inputs, self.prediction_config, model
        )
        processor.validate()
        processor.timestamps = [86400, 172800, 259200]

        inputs = None
        processor = CultivationHistoricalPreprocessor(
            timestamps, "d", steps, inputs, self.prediction_config, model
        )

        with self.assertRaises(InvalidInputsException) as ex:
            processor.validate()
            self.assertEqual(
                ex.exception.message,
                "No Inputs provided.",
            )

        inputs = [10, 20, 30]
        processor = CultivationHistoricalPreprocessor(
            timestamps, "d", steps, inputs, self.prediction_config, model
        )

        with self.assertRaises(InvalidInputsException) as ex:
            processor.validate()
            self.assertEqual(
                ex.exception.message,
                "Inputs must be a dictionary of lists",
            )

        inputs = {
            "var1": 30,
            "var2": 20,
            "var3": [1, 2, 3],
            "var4": [40, 50, 60],
        }
        processor = CultivationHistoricalPreprocessor(
            timestamps, "d", steps, inputs, self.prediction_config, model
        )

        with self.assertRaises(InvalidInputsException) as ex:
            processor.validate()
            self.assertEqual(
                ex.exception.message,
                "All input values must be lists",
            )

        inputs = {
            "var1": [10, "20", 30],
            "var2": [20, 20, 20],
            "var3": [1, 2, 3],
            "var4": [40, 50, 60],
        }
        processor = CultivationHistoricalPreprocessor(
            timestamps, "d", steps, inputs, self.prediction_config, model
        )
        with self.assertRaises(InvalidInputsException) as ex:
            processor.validate()
            self.assertEqual(
                ex.exception.message,
                "All values of input var1 must be valid numeric values",
            )

        inputs = {
            "var1": [10, 20, 30],
            "var2": [20, 20, 20],
            "var4": [40, 50, 60],
        }
        processor = CultivationHistoricalPreprocessor(
            timestamps, "d", steps, inputs, self.prediction_config, model
        )
        with self.assertRaises(InvalidInputsException) as ex:
            processor.validate()
            self.assertEqual(
                ex.exception.message,
                "Input var3 is a W Variable, so it must be provided",
            )

    def test_historical_steps(self):
        model = self.model
        inputs = {
            "var1": [10, 10, 10],
            "var2": [20, 20, 20],
            "var3": [1, 2, 3],
            "var4": [40],
        }
        timestamps = [1, 2, 3]
        steps = [0, 1, 2]

        steps = None
        processor = CultivationHistoricalPreprocessor(
            timestamps, "d", steps, inputs, self.prediction_config, model
        )

        with self.assertRaises(InvalidStepsException) as ex:
            processor.validate()
            self.assertEqual(
                ex.exception.message,
                "Steps must be a list of numbers",
            )

        steps = [0, 1, 2, 3]
        processor = CultivationHistoricalPreprocessor(
            timestamps, "d", steps, inputs, self.prediction_config, model
        )

        with self.assertRaises(InvalidStepsException) as ex:
            processor.validate()
            self.assertEqual(
                ex.exception.message,
                "Steps must have the same length as timestamps",
            )

        steps = [1, 2, 3]
        processor = CultivationHistoricalPreprocessor(
            timestamps, "d", steps, inputs, self.prediction_config, model
        )

        with self.assertRaises(InvalidStepsException) as ex:
            processor.validate()
            self.assertEqual(
                ex.exception.message,
                "Steps must start at 0",
            )

        steps = [0, 2, 1]
        processor = CultivationHistoricalPreprocessor(
            timestamps, "d", steps, inputs, self.prediction_config, model
        )

        with self.assertRaises(InvalidStepsException) as ex:
            processor.validate()
            self.assertEqual(
                ex.exception.message,
                "Steps must be in ascending order",
            )

    def test_convert_to_request(self):
        model = self.model
        inputs = {
            "var4": [40],
            "var1": [10],
            "var3": [0.2, 0.6, 0.6, 0.1],
            "var2": [20],
        }
        timestamps = [0, 1, 2, 3]

        processor = CultivationPropagationPreprocessor(
            timestamps, "h", inputs, self.prediction_config, model
        )

        processor.validate()
        request = processor.format()

        self.assertEqual(request[0]["instances"][0][0]["values"], [10])  # var1
        self.assertEqual(request[0]["instances"][0][0]["timestamps"], [0])
        self.assertEqual(request[0]["instances"][0][1]["values"], [20])  # var2
        self.assertEqual(
            request[0]["instances"][0][2]["values"], [0.2, 0.6, 0.6, 0.1]
        )  # var3
        self.assertEqual(
            request[0]["instances"][0][2]["timestamps"], [0, 3600, 7200, 10800]
        )

    def test_convert_to_request_historical(self):
        model = self.model
        inputs = {
            "var1": [10, 10, 10],
            "var2": [20, 20, 20],
            "var3": [1, 2, 3],
            "var4": [40],
        }
        timestamps = [1, 2, 3]
        steps = [0, 1, 2]

        processor = CultivationHistoricalPreprocessor(
            timestamps, "d", steps, inputs, self.prediction_config, model
        )

        processor.validate()
        request = processor.format()

        self.assertEqual(request[0]["instances"][0][0]["values"], [10, 10, 10])
        self.assertEqual(
            request[0]["instances"][0][0]["timestamps"], [86400, 172800, 259200]
        )
        self.assertEqual(request[0]["instances"][0][0]["steps"], [0, 1, 2])
        self.assertEqual(request[0]["instances"][0][1]["values"], [20, 20, 20])
        self.assertEqual(request[0]["instances"][0][2]["values"], [1, 2, 3])


class TestResults(unittest.TestCase):
    def setUp(self):
        self.client = Mock()

    def test_results(self):
        class DummyEntity(BaseModel):
            id: int
            name: str

        class DummyRequests:
            def list(self, offset, limit, query_params):
                # creates a dummy fetch function that returns 15 elements
                return [
                    DummyEntity(id=i, name=f"Name {i}")
                    for i in range(offset, offset + limit)
                    if i < 15
                ], 15

        results = Result[DummyEntity](
            offset=0,
            limit=5,
            query_params={},
            requests=DummyRequests(),
        )

        self.assertEqual(len(results), 15)

        # tests the first 5 items
        self.assertEqual(next(results).id, 0)
        self.assertEqual(next(results).id, 1)
        self.assertEqual(next(results).id, 2)
        self.assertEqual(next(results).name, "Name 3")
        next(results)

        # tests that the fetch is called and that the client is assigned
        self.assertEqual(next(results).id, 5)

        # test that there are still 9 items available using the list function
        all_results = list(results)
        self.assertEqual(len(all_results), 9)

        # tests the StopIteration exception
        self.assertRaises(StopIteration, next, results)
