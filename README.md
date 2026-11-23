# NeuroScope · Brain MRI analysis

NeuroScope is a local research application for brain tumor detection, segmentation, localization, and Grad-CAM on preprocessed MRI scans.

| Deliverable | Contents |
| --- | --- |
| [`Code/`](Code/) | Four Colab notebooks for data preparation, training, evaluation, and explainability; shared Python code; and training tests. |
| [`Application/Client/`](Application/Client/) | React and TypeScript interface built with Vite. |
| [`Application/Server/`](Application/Server/) | FastAPI inference service, SQLite history, and API tests. |
| [`Application/Models/`](Application/Models/) | EfficientNet-B0 and SegResNet checkpoints, model cards, training configurations, and run summaries. |
| [`Application/SampleCases/`](Application/SampleCases/) | Two BraTS 2021 cases, each with four MRI volumes and a reference segmentation. See the [data attribution](Application/SampleCases/README.md). |

To run the delivered app with the included models and sample cases, follow the [setup guide](Application/README.md). Colab and retraining are not required. To reproduce the training workflow, follow the [notebook guide](Code/README.md). The [verification record](Application/VERIFICATION.md) documents the tested workflows and hardware.

Keep `Code/` and `Application/` together: the app imports the shared inference code from `Code/brats/`. Uploaded scans, job history, and generated reports stay in local storage outside Git. NeuroScope is intended for research use and is not a diagnostic medical device.
