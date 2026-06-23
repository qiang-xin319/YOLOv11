# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f"{m._get_name()}.onnx"
    torch.onnx.export(m, x, f)
    os.system(f"onnxslim {f} {f} && open {f}")  # pip install onnxslim
    ```
"""

from .block import  (
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    CIB,
    DFL,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Attention,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Proto,
    RepC3,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
    Bi_FPN,
    SEAM,
    C2PSA_SEAM,
    LSKA, C2PSA_LSKA,MLCA,C2f_MLCA,deformable_LKA_Attention, C2PSA_DLKA,EMA, C2PSA_EMA,MultiDilatelocalAttention,C2PSA_MSDA,SELayerV1, C2PSA_SENetV1
    ,BiLevelRoutingAttention,C2PSA_Biformer,DySample,DAttentionBaseline, C2PSA_DAT,FocalModulation,C2PSA_TripleAttention,TripletAttention,iRMB, C2PSA_iRMB
,C3k2_FasterBlock,C3k2_ContextGuidedBlock,ContextGuidedBlock_Down,C2PSA_SENetV2, SELayerV2, SPPFSENetV2,C3k2_simam,StripPooling,C3k2_SP,ACmix,C2PSA_ACmix
,GSConv,VoVGSCSP,Fusion)
# ,WindowAttention_ASSA,C3k2_WAtt_ASSA
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SpatialAttention,
    AKConv,
    WTConv2d,
    C3k2_WTConv,
    C3k2_ODConv,
    ODConv2d,
    DynamicConv,
    C3k2_DynamicConv,ADown,C3k2_RepVGG, RCSOSA,SAConv2d, C3k2_SAConv,Down_wt,C3k2_Faster,C3k2_WT,C3k2_SC,ScConv,C3k2_RCM,RCM,C3k2_GhostModule,LAE,C3k2_MSBlcok
  ,C3k2_PPA
)
#
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment, WorldDetect, v10Detect
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)
from .attention import *
__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C3k2",
    "SCDown",
    "C2fPSA",
    "C2PSA",
    "C2fAttn",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
    "WorldDetect",
    "v10Detect",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "AConv",
    "ELAN1",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
)
