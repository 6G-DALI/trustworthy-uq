# trustworthy-uq --- MLOps User Documentation

## 1\. Installation

`trustworthy-uq` is a Python package for the 2D CSI-localization
uncertainty-quantification workflows used in the 6G-DALI work.

The current package is designed for Python 3.10+.

### 1.1 Installation from GitHub

The package can be installed directly from the 6G-DALI GitHub repository.

Install the package with:

python -m pip install git+https://github.com/6G-DALI/trustworthy-uq.git



This installs the latest version available in the repository.

To install a specific version or Git commit, the corresponding tag or commit can be specified, for example:

python -m pip install git+https://github.com/6G-DALI/trustworthy-uq.git@v0.1.0



After installation, the package can be imported normally:

import trustworthy\_uq





For development, where changes to the package source need to be reflected immediately, clone the repository and install it in editable mode:

git clone https://github.com/6G-DALI/trustworthy-uq.git

cd trustworthy-uq

python -m pip install -e .



This allows modifications to the local package source to be used without reinstalling the package.



### 1.2 Verify the installation

Run:

``` bash
python -c "import trustworthy\_uq; print(trustworthy\_uq.\_\_version\_\_)"
```

Expected output for the current release:

``` text
0.1.0
```

You can also check where Python loaded the package from:

``` bash
python -c "import trustworthy\_uq; print(trustworthy\_uq.\_\_file\_\_)"
```

### 1.3 Dependencies

The package declares these dependencies:

* NumPy
* pandas
* PyTorch
* Lightning
* Plotly
* Kaleido

They are installed automatically by:

``` bash
python -m pip install -e .
```

Kaleido is used for static Plotly image export. HTML plots can still be
produced without successful PNG export.

\---

# 2\. What the package does

The package provides a ready-to-use workflow for adding uncertainty
information to an existing 2D CSI localization model.

The current supported localization backbone is **AttentionDenseNet
(ADN)**.

The package supports two uncertainty methods:

1. **CQR --- one-sided multi-SLA conformalized prediction**
2. **Adaptive SCP (aSCP) --- adaptive scale prediction followed by
conformal calibration**

The package is intentionally focused on the current 2D localization use
case rather than being a fully generic uncertainty-quantification
framework.

The typical workflow is:

``` text
CSI data
   |
   v
ADN localization model
   |
   +----------------------------+
   |                            |
   v                            v
 CQR head                    aSCP scale model
   |                            |
   v                            v
Conformal calibration       Conformal calibration
   |                            |
   +-------------+--------------+
                 |
                 v
       Prediction + uncertainty radius
                 |
                 v
      SLA coverage / breach metrics
```

The ADN produces the point localization.

The UQ component produces a radius around that point. The radius is
associated with a requested target coverage/SLA level such as 90%, 95%,
or 99%.

\---

# 3\. Important terminology for MLOps users

The package exposes a small number of concepts that are useful when
integrating it into an MLOps system.

### ADN

The existing localization model.

Input:

``` text
CSI sample
```

Output:

``` text
(x, y) localization
```

The ADN itself provides the point prediction.

### SLA level / target coverage

A requested probability of covering the true position.

Examples:

``` python
\[0.90, 0.95, 0.99]
```

These correspond to:

``` text
90% target coverage
95% target coverage
99% target coverage
```

The package can process several SLA levels in one workflow.

### Radius

The uncertainty region is represented as a circle centered at the ADN
prediction.

For example:

``` text
ADN prediction
      +
      |
   radius
<---------->
```

A test sample is considered covered when its true position falls within
the predicted radius.

### Coverage

The fraction of test samples whose true location is inside the predicted
uncertainty region.

For example:

``` text
true\_coverage = 0.956
```

means approximately 95.6% of the evaluated samples were covered.

### Breach rate

The fraction not covered:

``` text
breach\_rate = 1 - true\_coverage
```

For example:

``` text
coverage    = 0.956
breach rate = 0.044
```

\---

# 4\. Package structure

The main package structure is:

``` text
trustworthy\_uq/
|
+-- localization/
|   +-- adn.py
|   +-- backbone.py
|   +-- cqr.py
|   +-- adaptive\_scp.py
|   +-- pipeline.py
|
+-- data/
|   +-- kul.py
|   +-- nomadic.py
|
+-- calibration/
|   +-- conformal.py
|
+-- evaluation/
|   +-- metrics.py
|   +-- visualization.py
|
+-- artifacts/
|   +-- io.py
|
+-- tracking/
|   +-- mlflow.py
|
+-- \_\_init\_\_.py
```

For normal MLOps integration, users generally only need:

``` python
from trustworthy\_uq import LocalizationUQ
```

The lower-level modules are available when more specialized control is
required.

\---

# 5\. Main entry point: LocalizationUQ

The main workflow class is:

``` python
from trustworthy\_uq import LocalizationUQ
```

Create an instance:

``` python
uq = LocalizationUQ(
    sla\_levels=\[0.90, 0.95, 0.99],
    seed=42,
    batch\_size=32,
    num\_workers=0,
)
```

### Constructor parameters

``` python
LocalizationUQ(
    backbone=None,
    device=None,
    sla\_levels=None,
    seed=42,
    batch\_size=32,
    num\_workers=0,
    split=(0.25, 0.45, 0.30),
)
```

\---

Parameter                           Meaning

\---

`backbone`                          Optional already-created ADN model

`device`                            PyTorch device; automatically
selects CUDA when available

`sla\_levels`                        SLA/coverage levels supported by
the workflow

`seed`                              Random seed used for dataset
splitting

`batch\_size`                        Default inference/training batch
size

`num\_workers`                       DataLoader workers

`split`                             Dataset proportions for
training/head, calibration, and
test
---

The default split is:

``` text
25% training/head
45% calibration
30% test
```

The split is performed by `split\_dataset()`.

\---

# 6\. Dataset handling

The current package supports the Nomadic/KUL CSI localization dataset
used by the experiments.

## 6.1 Single scenario

``` python
dataset = uq.build\_scenario\_dataset(
    data\_dir="data/nomadic\_dataset/ULA\_lab\_LoS",
    scenario\_id=0,
)
```

The default configuration uses:

``` text
4 users
240 samples per user
```

These can be changed:

``` python
dataset = uq.build\_scenario\_dataset(
    data\_dir=DATA\_DIR,
    scenario\_id=0,
    num\_users=4,
    num\_samples=240,
)
```

## 6.2 Multiple scenarios

``` python
dataset = uq.build\_pooled\_dataset(
    data\_dir=DATA\_DIR,
    scenario\_ids=range(6),
)
```

This creates one combined dataset from the requested scenarios.

## 6.3 Dataset splitting

``` python
train\_dataset, calibration\_dataset, test\_dataset = uq.split\_dataset(dataset)
```

With the default split:

``` text
25% -> training/head
45% -> calibration
30% -> test
```

The same split must be used consistently when
training/calibrating/evaluating a workflow.

\---

# 7\. ADN model workflows

ADN is the localization backbone.

## 7.1 Load an existing ADN model

``` python
uq.load\_adn(
    "path/to/best\_model.ckpt"
)
```

After this:

``` python
uq.backbone
```

contains the loaded model.

The package does not require the ADN checkpoint to have been trained by
`trustworthy-uq`.

Any compatible ADN checkpoint can be loaded.

## 7.2 Train a new ADN model

``` python
result = uq.train\_adn(
    data\_dir=DATA\_DIR,
    output\_dir="output/adn",
    batch\_size=32,
    num\_workers=0,
    num\_users=4,
    num\_samples=240,
    max\_epochs=200,
    seed=42,
)
```

The function returns a dictionary containing the trained model and
checkpoint information.

Typical fields:

``` python
result\["model"]
result\["checkpoint\_path"]
result\["runtime\_seconds"]
```

The best checkpoint is saved by Lightning under the specified output
directory.

Example:

``` text
output/
└── adn/
    └── checkpoints/
        └── best\_adn\_backbone.ckpt
```

The package does not require an external model registry.

The MLOps platform can copy/register the returned checkpoint path using
its own artifact/model-management mechanism.

\---

# 8\. CQR workflow

CQR is the main multi-SLA uncertainty workflow.

The important operational point is:

**The CQR head is trained once for the configured SLA levels. It is not
retrained for every SLA calibration.**

For example:

``` python
SLA\_LEVELS = \[0.90, 0.95, 0.99]
```

creates one CQR head containing three SLA-specific radius outputs.

Calibration is then performed using the same trained head.

\---

# 9\. Load an existing CQR head

First load the ADN:

``` python
uq.load\_adn(ADN\_CHECKPOINT)
```

Then load the CQR checkpoint:

``` python
uq.load\_cqr(
    CQR\_CHECKPOINT
)
```

The package obtains the configured SLA levels from the checkpoint.

\---

# 10\. Train a new CQR head

A new CQR head can be trained on top of an already available ADN:

``` python
result = uq.train\_cqr(
    train\_dataset,
    validation\_dataset=calibration\_dataset,
    sla\_levels=\[0.90, 0.95, 0.99],
    max\_epochs=100,
    lr=1e-3,
    weight\_decay=1e-5,
    output\_dir="output/cqr",
)
```

The backbone is frozen during CQR-head training.

The returned dictionary contains:

``` python
result\["model"]
result\["checkpoint\_path"]
result\["runtime\_seconds"]
result\["sla\_levels"]
```

When `output\_dir` is supplied, the best checkpoint is saved under:

``` text
output/cqr/
└── checkpoints/
    └── best\_cqr\_head.ckpt
```

\---

# 11\. CQR calibration

After loading or training the CQR head:

``` python
calibration\_result = uq.calibrate\_cqr(
    calibration\_dataset,
    sla\_levels=\[0.90, 0.95, 0.99],
    save\_path="output/cqr/cqr\_calibration.json",
)
```

Calibration determines the conformal adjustment needed for each
requested SLA.

The same trained CQR head can therefore be calibrated for:

``` python
\[0.90]
```

or:

``` python
\[0.90, 0.95, 0.99]
```

without retraining the head.

The calibration artifact contains the resulting conformal adjustment
values.

\---

# 12\. aSCP workflow

Adaptive SCP is a second supported uncertainty workflow.

Operationally, it has two stages:

1. Train an adaptive scale model on top of the frozen ADN.
2. Apply conformal calibration to the normalized localization errors.

The adaptive scale model estimates how large the expected localization
error is for each input.

\---

# 13\. Train a new aSCP model

``` python
result = uq.train\_adaptive\_scp(
    train\_dataset,
    validation\_dataset=calibration\_dataset,
    max\_epochs=60,
    lr=1e-3,
    weight\_decay=1e-5,
    output\_dir="output/ascp",
)
```

The ADN backbone is frozen.

The trained scale model is saved as:

``` text
output/ascp/
└── checkpoints/
    └── best\_ascp\_scale.ckpt
```

The returned result contains:

``` python
result\["model"]
result\["checkpoint\_path"]
result\["runtime\_seconds"]
```

\---

# 14\. Load an existing aSCP model

``` python
uq.load\_ascp(
    "path/to/best\_ascp\_scale.ckpt"
)
```

This requires a compatible ADN backbone to already be loaded.

\---

# 15\. aSCP calibration

After loading or training the scale model:

``` python
result = uq.calibrate\_adaptive\_scp(
    calibration\_dataset,
    sla\_levels=\[0.90, 0.95, 0.99],
    save\_path="output/ascp/ascp\_calibration.json",
)
```

As with CQR, calibration can be performed for one or multiple SLA
levels.

The trained scale model itself is not retrained when changing the
requested SLA level.

\---

# 16\. Single-sample prediction

For a raw CSI sample:

``` python
result = uq.predict(
    samples=X,
    sla\_levels=\[0.90, 0.95, 0.99],
    y\_true=y,
    method="CQR",
)
```

The result contains:

``` python
result\["method"]
result\["sla\_levels"]
result\["point\_prediction"]
result\["radius\_mm"]
result\["error\_mm"]
result\["covered"]
```

For example:

``` text
point\_prediction
    -> ADN x,y prediction

radius\_mm
    -> uncertainty radius for each SLA

error\_mm
    -> actual localization error, when y\_true is supplied

covered
    -> whether the true position falls within each radius
```

The same API is used for aSCP:

``` python
result = uq.predict(
    samples=X,
    sla\_levels=\[0.90, 0.95, 0.99],
    y\_true=y,
    method="aSCP",
)
```

\---

# 17\. Test-set evaluation

For full test-set evaluation:

``` python
metrics, predictions = uq.evaluate(
    test\_dataset,
    sla\_levels=\[0.90, 0.95, 0.99],
    method="CQR",
    output\_dir="output/cqr\_evaluation",
)
```

The first returned object is a compact metrics DataFrame.

The second is the per-sample prediction DataFrame.

## Main metrics

For CQR, the metrics include:

* method
* target assurance
* alpha
* calibration size
* test size
* conformal adjustment (`qhat\_mm`)
* mean localization error
* median localization error
* 90th-percentile localization error
* 95th-percentile localization error
* true coverage
* breach rate
* mean base radius
* mean final radius

For aSCP, the corresponding radius statistics are reported as adaptive
interval-radius metrics.

\---

# 18\. Understanding the CQR radius fields

CQR reports two radius values.

### Base radius

`mean\_base\_radius\_mm`

This is the mean radius directly produced by the learned CQR head before
conformal calibration.

### Final radius

`mean\_final\_radius\_mm`

This is the radius after adding the conformal calibration adjustment.

Conceptually:

``` text
final radius
    =
learned CQR radius
    +
conformal adjustment
```

The final radius is the one used to determine whether a sample is
covered.

Therefore, when assessing the deployed UQ behavior, use:

``` text
true\_coverage
breach\_rate
mean\_final\_radius\_mm
```

rather than the base radius alone.

\---

# 19\. Understanding aSCP radius fields

For aSCP, the learned model produces an adaptive scale for each sample.

The calibrated radius is conceptually:

``` text
final radius
    =
predicted scale
    ×
conformal calibration factor
```

This means that the radius can vary substantially between samples.

The same SLA can therefore result in different radii for different CSI
inputs.

\---

# 20\. JSON artifacts

When `output\_dir` is supplied to `evaluate()` or `run\_experiment()`, the
package produces machine-readable JSON artifacts.

These are intended to make integration with an MLOps GUI
straightforward.

## 20.1 Whole-test-set result

``` text
uq\_test\_results.json
```

This is the main compact UQ result for an experiment.

It contains:

``` text
method
task
num\_test\_samples
model
calibration
sla\_results
```

The `sla\_results` section contains one entry for each requested SLA.

Typical information includes:

``` text
target assurance
alpha
test size
qhat
mean error
median error
P90 error
P95 error
true coverage
breach rate
mean radius
```

This file is the primary JSON artifact that an MLOps GUI should consume
for experiment-level UQ metrics.

\---

# 21\. Single-sample JSON artifact

When single-sample analysis is enabled:

``` text
uq\_single\_sample.json
```

is generated.

It contains:

``` text
method
task
sample\_index
model
point\_prediction
true\_position
sla\_results
```

Each SLA entry contains information such as:

``` text
target coverage
radius
covered
localization error
base radius
conformal adjustment
```

This is suitable for a GUI panel showing:

``` text
Sample
  |
  +-- ADN prediction
  |
  +-- true location
  |
  +-- SLA 90% -> radius / covered
  +-- SLA 95% -> radius / covered
  +-- SLA 99% -> radius / covered
```

\---

# 22\. Calibration JSON artifact

CQR:

``` text
cqr\_calibration.json
```

aSCP:

``` text
ascp\_calibration.json
```

These files contain the conformal calibration parameters.

They are useful for reproducing or loading calibration state.

They are not intended to replace `uq\_test\_results.json` as the main
experiment summary.

For an MLOps GUI, the recommended hierarchy is:

``` text
uq\_test\_results.json
    -> main experiment metrics

uq\_single\_sample.json
    -> individual prediction/UQ information

cqr\_calibration.json
ascp\_calibration.json
    -> calibration state
```

\---

# 23\. CSV artifacts

Evaluation also creates detailed tabular artifacts.

For CQR:

``` text
cqr\_metrics.csv
cqr\_predictions.csv
```

For aSCP:

``` text
adaptive\_scp\_metrics.csv
adaptive\_scp\_predictions.csv
```

The metrics CSV contains one row per SLA level.

The predictions CSV contains one row per:

``` text
test sample × SLA level
```

It therefore contains substantially more information than the compact
JSON result.

For example, the prediction CSV contains:

``` text
pred\_x
pred\_y
true\_x
true\_y
euclidean\_error\_mm
base\_radius\_mm
qhat\_mm
final\_radius\_mm
covered
```

These files are useful for detailed analysis or downstream
visualization.

\---

# 24\. Plot artifacts

The package generates Plotly visualizations.

Typical outputs include:

``` text
cqr\_coverage\_vs\_target.html
cqr\_coverage\_vs\_target.png

cqr\_mean\_radius\_vs\_target.html
cqr\_mean\_radius\_vs\_target.png
```

For aSCP the prefix is:

``` text
adaptive\_scp\_
```

The HTML versions are particularly suitable for direct embedding in a
web-based MLOps interface.

The PNG files are useful for reports or static dashboards.

\---

# 25\. Single-sample visualization

A single prediction can be visualized using:

``` python
uq.plot\_prediction(
    prediction=\[x\_pred, y\_pred],
    true\_position=\[x\_true, y\_true],
    radius=radius,
    sla\_level=0.95,
    method="CQR",
    output\_path="output/single\_sample",
)
```

This produces a plot showing:

* predicted position
* true position
* uncertainty circle

Use `plot\_prediction()` when the prediction has already been computed.

Use `plot\_sample()` when raw CSI data must first be passed through the
model.

\---

# 26\. Recommended MLOps integration

The package does not need to contain an HTTP/API server.

A recommended architecture is:

``` text
MLOps GUI
    |
    v
MLOps API/service
    |
    v
trustworthy-uq
    |
    +--> dataset
    +--> ADN checkpoint
    +--> CQR/aSCP checkpoint
    |
    +--> calibration
    +--> inference
    +--> evaluation
    |
    v
Artifacts
```

The API layer can call the package and expose its outputs.

For example:

``` python
metrics, predictions = uq.evaluate(...)
```

The API can then return:

``` json
{
  "sla\_results": {
    "0.95": {
      "true\_coverage": 0.956,
      "breach\_rate": 0.044,
      "mean\_final\_radius\_mm": 22.80
    }
  }
}
```

Similarly, the API can return the contents of:

``` text
uq\_single\_sample.json
```

for a GUI single-prediction page.

There is therefore no need to add web-server functionality to the
package itself.

\---

# 27\. Recommended API operations

A simple MLOps service can expose operations conceptually equivalent to:

### Load model

``` text
POST /uq/model/load
```

Inputs:

``` text
ADN checkpoint
CQR/aSCP checkpoint
method
```

### Calibrate

``` text
POST /uq/calibrate
```

Inputs:

``` text
calibration dataset
method
SLA levels
```

Output:

``` text
calibration JSON
```

### Evaluate

``` text
POST /uq/evaluate
```

Inputs:

``` text
test dataset
method
SLA levels
```

Output:

``` text
uq\_test\_results.json
```

### Single prediction

``` text
POST /uq/predict
```

Inputs:

``` text
CSI sample
method
SLA levels
```

Output:

``` text
prediction
radius for each SLA
```

### Single-sample visualization

``` text
GET /uq/sample/{id}/plot
```

The actual endpoint naming is outside the package and can be chosen by
the MLOps team.

\---

# 28\. One-call experiment workflow

For reproducing the complete experiment workflow, use:

``` python
uq.run\_experiment(...)
```

This is the easiest function for an experiment runner or MLOps
orchestration layer.

Example using existing models:

``` python
uq.run\_experiment(
    data\_dir=DATA\_DIR,
    method="CQR",
    eval\_mode="pooled",
    scenario\_ids=range(6),
    adn\_checkpoint=ADN\_CHECKPOINT,
    cqr\_checkpoint=CQR\_CHECKPOINT,
    train\_new\_adn=False,
    train\_new\_cqr=False,
    train\_new\_ascp=False,
    sla\_levels=\[0.90, 0.95, 0.99],
    output\_dir="output/experiment",
    enable\_single\_sample=True,
    single\_sample\_index=0,
    single\_sample\_sla=0.95,
)
```

For aSCP:

``` python
uq.run\_experiment(
    data\_dir=DATA\_DIR,
    method="aSCP",
    eval\_mode="pooled",
    scenario\_ids=range(6),
    adn\_checkpoint=ADN\_CHECKPOINT,
    ascp\_checkpoint=ASCP\_CHECKPOINT,
    train\_new\_adn=False,
    train\_new\_ascp=False,
    sla\_levels=\[0.90, 0.95, 0.99],
    output\_dir="output/experiment",
    enable\_single\_sample=True,
)
```

\---

# 29\. Training from scratch through run\_experiment

The one-call workflow can also train models.

### New CQR head

``` python
uq.run\_experiment(
    data\_dir=DATA\_DIR,
    method="CQR",
    adn\_checkpoint=ADN\_CHECKPOINT,
    train\_new\_cqr=True,
    sla\_levels=\[0.90, 0.95, 0.99],
    output\_dir="output/new\_cqr",
)
```

The new CQR checkpoint is saved below the experiment output directory.

### New aSCP model

``` python
uq.run\_experiment(
    data\_dir=DATA\_DIR,
    method="aSCP",
    adn\_checkpoint=ADN\_CHECKPOINT,
    train\_new\_ascp=True,
    sla\_levels=\[0.90, 0.95, 0.99],
    output\_dir="output/new\_ascp",
)
```

The new aSCP checkpoint is saved below the experiment output directory.

### New ADN

The workflow can also train an ADN:

``` python
uq.run\_experiment(
    data\_dir=DATA\_DIR,
    method="CQR",
    train\_new\_adn=True,
    train\_new\_cqr=True,
    sla\_levels=\[0.90, 0.95, 0.99],
    output\_dir="output/full\_training",
)
```

The package returns the trained models/checkpoint paths through the
underlying training functions, while the MLOps system can register or
copy those checkpoints as required.

\---

# 30\. Model artifact management

The package does not require a model registry.

For MLOps integration, the recommended approach is:

``` text
Model registry / artifact store
            |
            v
        checkpoint
            |
            v
trustworthy-uq.load\_adn()
trustworthy-uq.load\_cqr()
trustworthy-uq.load\_ascp()
```

For newly trained models:

``` text
trustworthy-uq training
        |
        v
checkpoint file
        |
        v
MLOps artifact/model registry
```

This keeps model lifecycle management separate from the UQ package.

\---

# 31\. MLflow integration

The package contains a small optional MLflow adapter:

``` python
from trustworthy\_uq.tracking.mlflow import log\_evaluation
```

It can log:

* metrics CSV
* predictions CSV
* generated figures

Example:

``` python
log\_evaluation(
    mlflow,
    metrics\_df=metrics,
    metrics\_csv=Path("output/cqr/cqr\_metrics.csv"),
    predictions\_csv=Path("output/cqr/cqr\_predictions.csv"),
    figures\_dir=Path("output/cqr"),
)
```

The package does not require MLflow for its core operation.

An MLOps environment can therefore use:

* MLflow
* another experiment tracker
* a custom artifact store
* direct filesystem/object-storage APIs

without changing the UQ computation itself.

\---

# 32\. Recommended production workflow

For an existing trained ADN and existing CQR model:

``` python
from trustworthy\_uq import LocalizationUQ

uq = LocalizationUQ(
    sla\_levels=\[0.90, 0.95, 0.99]
)

uq.load\_adn(ADN\_CHECKPOINT)

uq.load\_cqr(CQR\_CHECKPOINT)

dataset = uq.build\_pooled\_dataset(
    DATA\_DIR,
    scenario\_ids=range(6)
)

train, calibration, test = uq.split\_dataset(dataset)

uq.calibrate\_cqr(
    calibration,
    sla\_levels=\[0.90, 0.95, 0.99],
    save\_path="output/cqr\_calibration.json"
)

metrics, predictions = uq.evaluate(
    test,
    sla\_levels=\[0.90, 0.95, 0.99],
    method="CQR",
    output\_dir="output/cqr"
)
```

For most MLOps applications, however, `run\_experiment()` is simpler
because it combines these steps.

\---

# 33\. What an MLOps developer needs to provide

For a standard inference/evaluation workflow, the MLOps system needs:

### Required

1. CSI dataset
2. compatible ADN checkpoint
3. compatible CQR or aSCP checkpoint
4. calibration data
5. requested SLA levels

### Optional

6. test dataset
7. output/artifact directory
8. MLflow or other tracking system
9. GUI/API layer

The MLOps developer does not need to implement the conformal calibration
mathematics.

\---

# 34\. What the package returns versus what it saves

There are two mechanisms.

### Python return values

Functions such as:

``` python
train\_cqr()
train\_adaptive\_scp()
calibrate\_cqr()
calibrate\_adaptive\_scp()
predict()
evaluate()
run\_experiment()
```

return Python objects/dictionaries/DataFrames.

### Filesystem artifacts

When an output directory or save path is supplied, the package can save:

``` text
model checkpoints
calibration JSON
test-result JSON
single-sample JSON
metrics CSV
prediction CSV
HTML plots
PNG plots
```

This separation is useful for MLOps because the API can consume returned
objects directly or retrieve persisted artifacts.

\---

# 35\. Error handling and prerequisites

The package intentionally keeps exception handling relatively
lightweight.

Before calling the UQ functions, ensure:

1. The ADN model is loaded.
2. The required CQR/aSCP model is loaded or trained.
3. Calibration has been performed before `predict()` or `evaluate()`.
4. Requested SLA levels exist in the trained CQR model.
5. The input CSI tensor has the expected shape.
6. The dataset uses the expected Nomadic/KUL format.

A common sequence is:

``` text
load ADN
   ↓
load/train CQR or aSCP
   ↓
calibrate
   ↓
predict/evaluate
```

\---

# 36\. Common issues

## `ModuleNotFoundError: trustworthy\_uq`

Usually the IDE is using a different Python interpreter.

Check:

``` python
import sys
print(sys.executable)
```

Then:

``` python
import trustworthy\_uq
print(trustworthy\_uq.\_\_file\_\_)
```

If necessary, reinstall into that interpreter:

``` bash
python -m pip install -e .
```

## Lightning import errors

The package requires a compatible Lightning installation.

Use:

``` bash
python -m pip install -U lightning
```

and verify:

``` python
import lightning
print(lightning.\_\_version\_\_)
```

## Missing checkpoint

Verify the path before loading:

``` python
from pathlib import Path

path = Path("path/to/checkpoint.ckpt")
print(path.exists())
```

## `KeyError` for an SLA level

For CQR, the requested SLA must be one of the SLA levels with which the
CQR head was trained.

For example, if the head was trained for:

``` python
\[0.90, 0.95, 0.99]
```

requesting:

``` python
\[0.975]
```

is not supported by that checkpoint.

Train a CQR head configured with the required SLA levels if that SLA is
needed.

## Prediction before calibration

`predict()` requires conformal calibration values.

The usual sequence is:

``` python
uq.calibrate\_cqr(...)
uq.predict(...)
```

or:

``` python
uq.calibrate\_adaptive\_scp(...)
uq.predict(..., method="aSCP")
```

\---

# 37\. Minimal CQR example

``` python
from trustworthy\_uq import LocalizationUQ

DATA\_DIR = "data/nomadic\_dataset/ULA\_lab\_LoS"

uq = LocalizationUQ(
    sla\_levels=\[0.90, 0.95, 0.99]
)

uq.load\_adn("best\_model.ckpt")
uq.load\_cqr("best\_cqr\_head.ckpt")

dataset = uq.build\_pooled\_dataset(
    DATA\_DIR,
    scenario\_ids=range(6)
)

train, calibration, test = uq.split\_dataset(dataset)

uq.calibrate\_cqr(
    calibration,
    sla\_levels=\[0.90, 0.95, 0.99],
    save\_path="output/cqr\_calibration.json"
)

metrics, predictions = uq.evaluate(
    test,
    sla\_levels=\[0.90, 0.95, 0.99],
    method="CQR",
    output\_dir="output/cqr"
)

print(metrics)
```

\---

# 38\. Minimal aSCP example

``` python
from trustworthy\_uq import LocalizationUQ

DATA\_DIR = "data/nomadic\_dataset/ULA\_lab\_LoS"

uq = LocalizationUQ(
    sla\_levels=\[0.90, 0.95, 0.99]
)

uq.load\_adn("best\_model.ckpt")
uq.load\_ascp("best\_ascp\_scale.ckpt")

dataset = uq.build\_pooled\_dataset(
    DATA\_DIR,
    scenario\_ids=range(6)
)

train, calibration, test = uq.split\_dataset(dataset)

uq.calibrate\_adaptive\_scp(
    calibration,
    sla\_levels=\[0.90, 0.95, 0.99],
    save\_path="output/ascp\_calibration.json"
)

metrics, predictions = uq.evaluate(
    test,
    sla\_levels=\[0.90, 0.95, 0.99],
    method="aSCP",
    output\_dir="output/ascp"
)

print(metrics)
```

\---

# 39\. Minimal MLOps-oriented workflow

For an MLOps service, the core sequence can be reduced to:

``` python
uq = LocalizationUQ(
    sla\_levels=\[0.90, 0.95, 0.99]
)

uq.load\_adn(adn\_checkpoint)

uq.load\_cqr(cqr\_checkpoint)

metrics, predictions = uq.run\_experiment(
    data\_dir=data\_dir,
    method="CQR",
    eval\_mode="pooled",
    scenario\_ids=range(6),
    adn\_checkpoint=adn\_checkpoint,
    cqr\_checkpoint=cqr\_checkpoint,
    sla\_levels=\[0.90, 0.95, 0.99],
    output\_dir=output\_dir,
    enable\_single\_sample=True,
)
```

The resulting output directory contains the main machine-readable
artifacts that can be exposed through the MLOps API.

\---

# 40\. API reference

## `LocalizationUQ`

### Data

``` python
build\_scenario\_dataset(
    data\_dir,
    scenario\_id,
    num\_users=4,
    num\_samples=240
)
```

``` python
build\_pooled\_dataset(
    data\_dir,
    scenario\_ids=range(6),
    num\_users=4,
    num\_samples=240
)
```

``` python
split\_dataset(dataset)
```

### ADN

``` python
load\_adn(checkpoint\_path)
```

``` python
train\_adn(data\_dir, \*\*kwargs)
```

### CQR

``` python
train\_cqr(
    train\_dataset,
    validation\_dataset=None,
    sla\_levels=None,
    max\_epochs=100,
    lr=1e-3,
    weight\_decay=1e-5,
    output\_dir=None
)
```

``` python
load\_cqr(checkpoint\_path)
```

``` python
calibrate\_cqr(
    calibration\_dataset,
    sla\_levels=None,
    save\_path=None
)
```

### aSCP

``` python
train\_adaptive\_scp(
    train\_dataset,
    validation\_dataset=None,
    max\_epochs=60,
    lr=1e-3,
    weight\_decay=1e-5,
    output\_dir=None
)
```

``` python
load\_ascp(checkpoint\_path)
```

``` python
calibrate\_adaptive\_scp(
    calibration\_dataset,
    sla\_levels=None,
    save\_path=None
)
```

### Inference

``` python
predict(
    samples,
    sla\_levels=None,
    y\_true=None,
    method="CQR",
    output\_path=None
)
```

### Evaluation

``` python
evaluate(
    test\_dataset,
    sla\_levels=None,
    method="CQR",
    output\_dir=None
)
```

### Visualization

``` python
plot\_sample(
    samples,
    y\_true,
    sample\_index=0,
    sla\_level=0.95,
    method="CQR",
    output\_path=None
)
```

``` python
plot\_prediction(
    prediction,
    true\_position,
    radius,
    sla\_level=0.95,
    method="CQR",
    output\_path=None
)
```

### Complete workflow

``` python
run\_experiment(
    data\_dir,
    method="CQR",
    eval\_mode="pooled",
    scenario\_id=0,
    scenario\_ids=range(6),
    adn\_checkpoint=None,
    cqr\_checkpoint=None,
    ascp\_checkpoint=None,
    train\_new\_adn=False,
    train\_new\_cqr=False,
    train\_new\_ascp=False,
    sla\_levels=None,
    output\_dir="output/unified\_conformal\_experiments",
    enable\_single\_sample=True,
    single\_sample\_index=0,
    single\_sample\_sla=0.95,
    \*\*train\_kwargs
)
```

\---

# 41\. Recommended artifact contract for the MLOps GUI

For integration, the simplest contract is:

``` text
Input
-----
dataset
ADN checkpoint
UQ checkpoint
method
SLA levels


Output
------
uq\_test\_results.json
uq\_single\_sample.json
cqr\_calibration.json / ascp\_calibration.json
metrics CSV
predictions CSV
HTML/PNG plots
```

The GUI does not need to understand the internals of conformal
prediction.

It can treat the package as a service that returns:

``` text
point prediction
+
uncertainty radius
+
target SLA
+
empirical coverage
+
breach rate
+
radius statistics
```

This is the intended abstraction boundary between the UQ implementation
and the MLOps layer.

\---

# 42\. Current scope

The current release is intentionally focused on:

* 2D CSI localization
* Nomadic/KUL dataset workflow
* ADN localization backbone
* one-sided multi-SLA CQR
* adaptive SCP with a learned scale model
* conformal calibration
* test-set coverage and breach evaluation
* single-sample UQ
* machine-readable artifacts
* optional MLflow artifact logging

It is not currently intended as a generic framework for arbitrary
regression models, arbitrary datasets, or arbitrary UQ algorithms.

The package structure, however, is designed so additional UQ methods and
classification workflows can be added later.

\---

# 43\. Operational summary

For an MLOps user, the complete lifecycle is:

``` text
1. Install package
       ↓
2. Load ADN checkpoint
       ↓
3. Load or train CQR/aSCP model
       ↓
4. Build/select dataset
       ↓
5. Calibrate using calibration data
       ↓
6. Run prediction/evaluation
       ↓
7. Retrieve JSON/CSV/plots
       ↓
8. Expose results through MLOps API/GUI
```

The MLOps layer is responsible for:

* model/data artifact storage
* authentication and API endpoints
* experiment tracking
* GUI presentation
* model lifecycle management

`trustworthy-uq` is responsible for:

* loading/training the supported models
* UQ calibration
* UQ prediction
* coverage/breach evaluation
* UQ-specific artifacts and visualizations

