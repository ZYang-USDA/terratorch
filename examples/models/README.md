## Models

This folder contains additional examples for working with the available backbones in TerraTorch.

All examples use the [Sen1Floods11 segmentation dataset](https://github.com/cloudtostreet/Sen1Floods11). We recommend first going through the segmentation task tutorial in TerraTorch:

Tutorial: `examples/segmentation/segmentation_sen1floods11.ipynb`

### Examples
This folder includes YAML configuration examples for different backbones (e.g. DOFA) and model factories (e.g. SMP Model Factory). These configs can be used with the TerraTorch CLI for fine-tuning:

```bash
terratorch fit -c path/to/config.yaml
```

The notebook `models_notebook.ipynb` walks you through downloading the dataset and running one of the provided YAML configurations.

Note:
- For Prithvi models, see `examples/segmentation/segmentation_sen1floods11.ipynb`.
- For TerraMind, see `examples/multimodal_data/multimodal_segmentation_sen1floods11.yaml`.