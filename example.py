# %%
import glodap

from PyESPER.lir import PyESPER_LIR
from PyESPER.mixed import PyESPER_Mixed
from PyESPER.nn import PyESPER_NN

data = glodap.atlantic()

L = slice(500, 1000)
latitude = data[L].latitude.values.tolist()
longitude = data[L].longitude.values.tolist()
depth = data[L].depth.values.tolist()
salinity = data[L].salinity.values.tolist()
temperature = data[L].temperature.values.tolist()
phosphate = data[L].phosphate.values.tolist()
nitrate = data[L].nitrate.values.tolist()
silicate = data[L].silicate.values.tolist()
oxygen = data[L].oxygen.values.tolist()
EstDates = data[L].year.values.tolist()

OutputCoordinates = {}
PredictorMeasurements = {}

OutputCoordinates.update({"longitude": longitude, "latitude": latitude, "depth": depth})

PredictorMeasurements.update(
    {
        "salinity": salinity,
        "temperature": temperature,
        "phosphate": phosphate,
        "nitrate": nitrate,
        "silicate": silicate,
        "oxygen": oxygen,
    }
)

MeasUncerts = {
    "sal_u": [0.001],
    "temp_u": [0.3],
    "phosphate_u": [0.14],
    "nitrate_u": [0.5],
    "silicate_u": [0.03],
    "oxygen_u": [0.025],
}


Path = "/Users/matthew/github/LarissaMDias/"

EstimatesNN, UncertaintiesNN = PyESPER_NN(
    ["DIC"],
    Path,
    OutputCoordinates,
    PredictorMeasurements,
    EstDates=EstDates,
    Equations=[1],
)
EstimatesLIR, UncertaintiesLIR, CoefficientsLIR = PyESPER_LIR(
    ["pH"],
    Path,
    OutputCoordinates,
    PredictorMeasurements,
    EstDates=EstDates,
    Equations=[1],
)
EstimatesMixed, UncertaintiesMixed = PyESPER_Mixed(
    ["phosphate", "TA"],
    Path,
    OutputCoordinates,
    PredictorMeasurements,
    VerboseTF=True,
    EstDates=EstDates,
    MeasUncerts=MeasUncerts,
    Equations=[1, 2],
)
print(EstimatesNN[90:100], UncertaintiesLIR[10:20], EstimatesMixed[0:10])
