
Encrypted Malware Traffic Detection via Flow Behavior Graph Adapting to Protocol Evolution

## Introduction
With the rapid evolution of cyber threats, malware increasingly exploits encrypted protocols for covert communication. The emergence of the QUIC protocol and widespread adoption of HTTP/3 have expanded the mainstream protocol combinations from [TCP-TLS] to [TCP-TLS, UDP-QUIC]. This enables malware to adopt more diverse evasion strategies at different lifecycle stages, thus rendering traditional detection methods insufficient for QUIC-enabled scenarios. To address this challenge, we propose QMTA, a multi-protocol malware traffic analysis framework that constructs flow behavior graph representations for infected hosts. Our approach models flow features from heterogeneous protocol stacks as graph nodes and their spatiotemporal correlations as edges, then leverages graph representation learning to generate low-dimensional embeddings that capture complex behavioral patterns across protocols. We construct a novel dataset containing QUIC-enhanced encrypted malware traffic through systematic simulation across critical lifecycle stages. Experimental results demonstrate that QMTA significantly outperforms five SOTA baselines in detecting both conventional and QUIC-enhanced encrypted malware traffic, achieving an AUC of 0.9412 and F1-score of 90.51\% on our novel dataset with robust generalization against unknown threats.

##  QMTA Overview

<p align="center">
  <img src="Attachments_Readme/IMG-Readme-20251128110524366.png" alt="QMTA Overview" width="80%"/>
</p>

## Hyperparameter Configuration

```
QMTA_Model(
  (gcn1): GCNConv(51, 128)
  (gcn2): GCNConv(128, 128)
  (gat1): GATConv(128, 32, heads=4)
  (gat2): GATConv(128, 32, heads=4)
  (global_conv): GraphConv(128, 64)
  (prototype_projection_head): Sequential(
    (0): Linear(in_features=128, out_features=128, bias=True)
    (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Linear(in_features=128, out_features=64, bias=True)
    (4): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=64, out_features=64, bias=True)
    (7): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (classification_head): Sequential(
    (0): Linear(in_features=128, out_features=64, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.3, inplace=False)
    (3): Linear(in_features=64, out_features=3, bias=True)
  )
  (dropout): Dropout(p=0.3, inplace=False)
)

================================================================================
Model Parameter Statistics:
================================================================================
gcn1.bias                                                    torch.Size([128])                       128 parameters
gcn1.lin.weight                                              torch.Size([128, 51])                 6,528 parameters
gcn2.bias                                                    torch.Size([128])                       128 parameters
gcn2.lin.weight                                              torch.Size([128, 128])               16,384 parameters
gat1.att_src                                                 torch.Size([1, 4, 32])                  128 parameters
gat1.att_dst                                                 torch.Size([1, 4, 32])                  128 parameters
gat1.bias                                                    torch.Size([128])                       128 parameters
gat1.lin.weight                                              torch.Size([128, 128])               16,384 parameters
gat2.att_src                                                 torch.Size([1, 4, 32])                  128 parameters
gat2.att_dst                                                 torch.Size([1, 4, 32])                  128 parameters
gat2.bias                                                    torch.Size([128])                       128 parameters
gat2.lin.weight                                              torch.Size([128, 128])               16,384 parameters
global_conv.lin_rel.weight                                   torch.Size([64, 128])                 8,192 parameters
global_conv.lin_rel.bias                                     torch.Size([64])                         64 parameters
global_conv.lin_root.weight                                  torch.Size([64, 128])                 8,192 parameters
prototype_projection_head.0.weight                           torch.Size([128, 128])               16,384 parameters
prototype_projection_head.0.bias                             torch.Size([128])                       128 parameters
prototype_projection_head.1.weight                           torch.Size([128])                       128 parameters
prototype_projection_head.1.bias                             torch.Size([128])                       128 parameters
prototype_projection_head.3.weight                           torch.Size([64, 128])                 8,192 parameters
prototype_projection_head.3.bias                             torch.Size([64])                         64 parameters
prototype_projection_head.4.weight                           torch.Size([64])                         64 parameters
prototype_projection_head.4.bias                             torch.Size([64])                         64 parameters
prototype_projection_head.6.weight                           torch.Size([64, 64])                  4,096 parameters
prototype_projection_head.6.bias                             torch.Size([64])                         64 parameters
prototype_projection_head.7.weight                           torch.Size([64])                         64 parameters
prototype_projection_head.7.bias                             torch.Size([64])                         64 parameters
classification_head.0.weight                                 torch.Size([64, 128])                 8,192 parameters
classification_head.0.bias                                   torch.Size([64])                         64 parameters
classification_head.3.weight                                 torch.Size([3, 64])                     192 parameters
classification_head.3.bias                                   torch.Size([3])                           3 parameters
================================================================================
Total Parameters: 111,043
Trainable Parameters: 111,043
Non-trainable Parameters: 0
================================================================================
```


| Category | Hyperparameter | Selection Range | Final Value | Description |
|----------|----------------|-----------------|-------------|-------------|
| **Model Architecture** | `hidden_dim` | [64, 128, 256] | **128** | GNN hidden layer dimension |
| | `projection_dim` | [32, 64, 128] | **64** | Contrastive learning projection dimension |
| | `dropout` | [0.2, 0.3, 0.5] | **0.3** | Dropout ratio |
| **Loss Weights** | `proto_weight` | [0.3, 0.4, 0.5] | **0.4** | Prototypical contrastive loss weight |
| | `instance_weight` | [0.1, 0.2, 0.3] | **0.2** | Instance contrastive loss weight |
| | `classification_weight` | [0.3, 0.4, 0.5] | **0.4** | Classification loss weight |
| **Contrastive Learning** | `temperature` | [0.05, 0.07, 0.1, 0.2] | **0.07** | Temperature coefficient for contrastive learning |
| | `proto_momentum` | [0.90, 0.95, 0.99] | **0.99** | Prototype momentum update coefficient |
| **Data Augmentation** | `noise_std` | [0.01, 0.05, 0.1] | **0.05** | Gaussian noise standard deviation |
| | `mask_ratio` | [0.05, 0.1, 0.15] | **0.05** | Feature masking ratio |
| | `dropout_ratio` | [0.05, 0.1, 0.15] | **0.1** | Feature dropout ratio |
| **Training Configuration** | `batch_size` | [16, 32, 64] | **32** (adaptive) | Batch size |
| | `learning_rate` | [0.0001, 0.001, 0.01] | **0.001** | Initial learning rate |
| | `weight_decay` | [1e-5, 1e-4, 1e-3] | **1e-4** | L2 regularization coefficient |
| | `epochs` | [100, 150, 200] | **150** | Number of training epochs |
| | `patience` | [20, 35, 50] | **35** | Early stopping patience |
| **Learning Rate Scheduling** | `scheduler_patience` | [5, 7, 10] | **7** | Learning rate decay patience |
| | `scheduler_factor` | [0.5, 0.7, 0.8] | **0.7** | Learning rate decay factor |
| | `gradient_clip` | [0.5, 1.0, 2.0] | **1.0** | Gradient clipping max norm |

## Development Environment

- Python 3.8
- PyTorch 2.0.1
- NetworkX 3.3
- Ubuntu 22.04

## Acknowledgments

We thank https://www.malware-traffic-analysis.net/ for providing the supporting data and tutorials.

We acknowledge two open-source QUIC-supported C2 frameworks: Merlin (https://github.com/Ne0nd0g/merlin) and Deimos (https://github.com/DeimosC2/DeimosC2).

We appreciate the WIDE and Tranco datasets for being open-source.