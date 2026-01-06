## Embedding Workflows

This folder contains examples for generating and working with embeddings in TerraTorch.

In standard TerraTorch workflows, downstream tasks are trained end-to-end on raw image data using an encoder–decoder pipeline. 
Embedding workflows decouple this process: With TerraTorch, you can extract EO embeddings from any supported backbone for your own data. These embeddings can then be analyzed, stored, and reused. 
Lightweight decoder-only pipelines can be trained directly on these precomputed embeddings.

### Examples
This folder includes:

- **Embedding Generation**:
    A demo notebook and sample YAML for easy embedding generation in TerraTorch: [`embedding_generation_burnscars.ipynb`](embedding_generation_burnscars.ipynb)  
  ([Open in Colab](https://colab.research.google.com/github/terrastackai/terratorch/blob/main/examples/embeddings/embedding_generation_burnscars.ipynb))

- **Embedding-Based Downstream Task**:
    A demo notebook and sample YAML for a segmentation task using precomputed embeddings: [`downstream_segmentation_burnscars.ipynb`](downstream_segmentation_burnscars.ipynb)  
  ([Open in Colab](https://colab.research.google.com/github/terrastackai/terratorch/blob/main/examples/embeddings/downstream_segmentation_burnscars.ipynb))

- **Manual Embedding Extraction via Backbone Registry**:
    A short demo showing how to use the TerraTorch Backbone Registry to build a custom pipeline and extract embeddings manually: [`embedding_generation_manual_backbone_registry.ipynb`](embedding_generation_manual_backbone_registry.ipynb)  
