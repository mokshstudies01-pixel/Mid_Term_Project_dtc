# Predictive Maintenance Failure Prediction

This project uses the AI4I 2020 Predictive Maintenance dataset to predict whether
industrial equipment will fail and, separately, which failure type is most
likely. The intended use is to help maintenance teams prioritize inspection;
these predictions should not replace operational safety procedures.

## Dataset

The project uses the public [AI4I 2020 Predictive Maintenance dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
from the UCI Machine Learning Repository. `train.py` downloads the CSV directly
from UCI, so the dataset does not need to be committed to this repository.
Internet access is required when training.

The notebook prepares the features, checks missing values and distributions,
examines correlations, and compares classifiers. The model inputs are air
temperature, process temperature, rotational speed, torque, tool wear, and
product type.

## Models and evaluation

The project trains:

- A binary classifier for `machine_failure`
- A multiclass classifier for `failure_type`

The notebook compares logistic regression and random forest for the binary
task, and tunes the random forest with cross-validation. On the validation
split, the recorded binary ROC-AUC is 0.9543 for the tuned random forest; the
multiclass accuracy is 0.9785. The multiclass classes are highly imbalanced,
so accuracy alone is not representative: the notebook's classification report
shows no validation recall for `multi_failure` and `other`.

The notebook creates test splits but reports validation metrics rather than
final held-out test metrics. Feature-importance analysis should also be added
before claiming the notebook's EDA is fully comprehensive.

## Environment setup

The serialized models were created with scikit-learn 1.8.0. Use Python 3.11 and
the dependencies in `requirements.txt`.

PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-notebook.txt
```

To run the notebook, start Jupyter after installing the dependencies:

```powershell
jupyter lab Untitled0.ipynb
```

## Train the models

From the project directory, with the virtual environment active and internet
access available:

```powershell
python train.py
```

This downloads the dataset, trains both models, and writes
`model_binary.bin` and `model_multiclass.bin` to the project directory.

## Run the prediction API locally

```powershell
python predict.py
```

The Flask development server listens on port 9696. Send a `POST` request to
`/predict` with the five numeric features and a product `type` (`L`, `M`, or
`H`):

```powershell
curl.exe -X POST http://localhost:9696/predict `
  -H "Content-Type: application/json" `
  -d '{"air_temperature_k":298.1,"process_temperature_k":308.6,"rotational_speed_rpm":1551,"torque_nm":42.8,"tool_wear_min":0,"type":"M"}'
```

The response contains `failure_probability` and `predicted_failure_type`.

## Run with Docker

Build and start the API from the project directory:

```powershell
docker build -t predictive-maintenance .
docker run --rm -p 9696:9696 predictive-maintenance
```

Then send the same request shown above. The Docker image serves the checked-in
model artifacts; run `python train.py` first if you want to regenerate them.

## Submission checklist

- [x] Problem description and dataset source
- [x] Notebook with data preparation, EDA, model comparison, and tuning
- [x] Training script and prediction web service
- [x] Dependency file and Dockerfile
- [ ] Add feature-importance analysis and report final test-set metrics
- [ ] Build and run the Docker image from a fresh checkout
- [ ] Include a screenshot/video showing a successful prediction, or a deployed
  service URL, as deployment evidence
- [ ] Submit the final GitHub repository URL and commit ID through the course
  submission form

Cloud deployment is not included. It is optional in the course guidance.
