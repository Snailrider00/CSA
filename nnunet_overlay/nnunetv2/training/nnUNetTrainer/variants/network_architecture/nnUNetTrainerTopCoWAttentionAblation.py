from __future__ import annotations

import pydoc

from copy import deepcopy

from typing import List, Tuple, Type, Union

import numpy as np

import torch

from torch import nn

from torch.nn.modules.conv import _ConvNd

from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures

from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim, get_matching_convtransp

from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder

from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks

from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerConnectivityAwareTopCoW import (
    nnUNetTrainerIMRClassWeightedSkeletonRecallNoMirroring1000,
)

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import nnUNetTrainerNoMirroring

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

class IdentitySkipGate(nn.Module):
    def forward(self, skip: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
        return skip

class ResidualSkipAttentionGate(nn.Module):
    """Attention U-Net style residual spatial gate for skip features."""

    def __init__(self, conv_op: Type[_ConvNd], channels: int, conv_bias: bool = True):
        super().__init__()
        self.proj = conv_op(channels * 2, 1, 1, 1, 0, bias=conv_bias)
        self.beta = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def forward(self, skip: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.proj(torch.cat((skip, decoder), dim=1)))
        return skip * (1 + self.beta * (gate - 0.5))

class ECA3D(nn.Module):
    """Lightweight channel recalibration for 3D features."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.beta = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.avg_pool(x).flatten(2).transpose(1, 2)
        weights = torch.sigmoid(self.conv(weights).transpose(1, 2)).view(x.shape[0], x.shape[1], 1, 1, 1)
        return x * (1 + self.beta * (weights - 0.5))

class UNetDecoderWithOptionalSkipAttention(nn.Module):
    def __init__(
        self,
        encoder: PlainConvEncoder,
        num_classes: int,
        n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
        deep_supervision: bool,
        use_skip_attention: bool,
        nonlin_first: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict | None = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict | None = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict | None = None,
        conv_bias: bool | None = None,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs

        stages = []
        transpconvs = []
        seg_layers = []
        attention_gates = []
        gate_cls = ResidualSkipAttentionGate if use_skip_attention else IdentitySkipGate
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(
                transpconv_op(
                    input_features_below,
                    input_features_skip,
                    stride_for_transpconv,
                    stride_for_transpconv,
                    bias=conv_bias,
                )
            )
            attention_gates.append(gate_cls(encoder.conv_op, input_features_skip, conv_bias=True)
                                   if use_skip_attention else gate_cls())
            stages.append(
                StackedConvBlocks(
                    n_conv_per_stage[s - 1],
                    encoder.conv_op,
                    2 * input_features_skip,
                    input_features_skip,
                    encoder.kernel_sizes[-(s + 1)],
                    1,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first,
                )
            )
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.attention_gates = nn.ModuleList(attention_gates)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            skip = self.attention_gates[s](skips[-(s + 2)], x)
            x = torch.cat((x, skip), dim=1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x
        seg_outputs = seg_outputs[::-1]
        return seg_outputs if self.deep_supervision else seg_outputs[0]

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            if not isinstance(self.attention_gates[s], IdentitySkipGate):
                output += np.prod([1, *skip_sizes[-(s + 1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)
        return output

class TopCoWAttentionUNet(AbstractDynamicNetworkArchitectures):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        use_skip_attention: bool,
        use_eca: bool,
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict | None = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict | None = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict | None = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
    ):
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        self.encoder = PlainConvEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_conv_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True,
            nonlin_first=nonlin_first,
        )
        self.eca = ECA3D(self.encoder.output_channels[-1]) if use_eca else nn.Identity()
        self.decoder = UNetDecoderWithOptionalSkipAttention(
            self.encoder,
            num_classes,
            n_conv_per_stage_decoder,
            deep_supervision,
            use_skip_attention=use_skip_attention,
            nonlin_first=nonlin_first,
        )

    def forward(self, x):
        skips = self.encoder(x)
        skips[-1] = self.eca(skips[-1])
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        eca_size = np.int64(self.encoder.output_channels[-1]) if not isinstance(self.eca, nn.Identity) else np.int64(0)
        return (
            self.encoder.compute_conv_feature_map_size(input_size)
            + self.decoder.compute_conv_feature_map_size(input_size)
            + eca_size
        )

class _TopCoWAttentionTrainerBase:
    @staticmethod
    def _build_attention_network(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool,
        use_skip_attention: bool,
        use_eca: bool,
    ) -> nn.Module:
        architecture_kwargs = deepcopy(configuration_manager.network_arch_init_kwargs)
        for key in configuration_manager.network_arch_init_kwargs_req_import:
            if architecture_kwargs[key] is not None:
                architecture_kwargs[key] = pydoc.locate(architecture_kwargs[key])
        architecture_kwargs["deep_supervision"] = enable_deep_supervision
        return TopCoWAttentionUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            use_skip_attention=use_skip_attention,
            use_eca=use_eca,
            **architecture_kwargs,
        )

class _TopCoWSingleAttentionCleanBuilder:
    @staticmethod
    def _build_single_attention_network(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool,
        use_skip_attention: bool,
        use_eca: bool,
    ) -> nn.Module:
        return _TopCoWAttentionTrainerBase._build_attention_network(
            plans_manager,
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            use_skip_attention=use_skip_attention,
            use_eca=use_eca,
        )

class nnUNetTrainerTopCoWSkipAttentionNoMirroring1000(
    _TopCoWSingleAttentionCleanBuilder,
    nnUNetTrainerNoMirroring,
):
    """Clean attention-only ablation: DiceCE baseline plus residual skip spatial gate."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
        self.oversample_foreground_percent = 0.33

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return _TopCoWSingleAttentionCleanBuilder._build_single_attention_network(
            plans_manager,
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            use_skip_attention=True,
            use_eca=False,
        )

class nnUNetTrainerTopCoWECANoMirroring1000(
    _TopCoWSingleAttentionCleanBuilder,
    nnUNetTrainerNoMirroring,
):
    """Clean attention-only ablation: DiceCE baseline plus bottleneck ECA channel attention."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
        self.oversample_foreground_percent = 0.33

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return _TopCoWSingleAttentionCleanBuilder._build_single_attention_network(
            plans_manager,
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            use_skip_attention=False,
            use_eca=True,
        )

class _TopCoWJointAttentionCleanBuilder:
    @staticmethod
    def _build_joint_attention_network(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool,
    ) -> nn.Module:
        return _TopCoWAttentionTrainerBase._build_attention_network(
            plans_manager,
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            use_skip_attention=True,
            use_eca=True,
        )

class nnUNetTrainerTopCoWJointAttentionNoMirroring1000(
    _TopCoWJointAttentionCleanBuilder,
    nnUNetTrainerNoMirroring,
):
    """Clean attention-only ablation: DiceCE baseline plus spatial-channel attention."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
        self.oversample_foreground_percent = 0.33

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return _TopCoWJointAttentionCleanBuilder._build_joint_attention_network(
            plans_manager, configuration_manager, num_input_channels, num_output_channels, enable_deep_supervision
        )

class nnUNetTrainerTopCoWCALECACleanNoMirroring1000(
    _TopCoWSingleAttentionCleanBuilder,
    nnUNetTrainerIMRClassWeightedSkeletonRecallNoMirroring1000,
):
    """Clean CAL plus bottleneck ECA ablation with standard foreground oversampling."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
        self.oversample_foreground_percent = 0.33

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return _TopCoWSingleAttentionCleanBuilder._build_single_attention_network(
            plans_manager,
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            use_skip_attention=False,
            use_eca=True,
        )

class nnUNetTrainerTopCoWCALSkipCleanNoMirroring1000(
    _TopCoWSingleAttentionCleanBuilder,
    nnUNetTrainerIMRClassWeightedSkeletonRecallNoMirroring1000,
):
    """Clean CAL plus residual skip spatial attention with standard oversampling."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
        self.oversample_foreground_percent = 0.33

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return _TopCoWSingleAttentionCleanBuilder._build_single_attention_network(
            plans_manager,
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            use_skip_attention=True,
            use_eca=False,
        )

class nnUNetTrainerTopCoWCALJointAttentionCleanNoMirroring1000(
    _TopCoWJointAttentionCleanBuilder,
    nnUNetTrainerIMRClassWeightedSkeletonRecallNoMirroring1000,
):
    """Clean full ablation: CAL plus spatial-channel attention, without changing sampling."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
        self.oversample_foreground_percent = 0.33

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return _TopCoWJointAttentionCleanBuilder._build_joint_attention_network(
            plans_manager, configuration_manager, num_input_channels, num_output_channels, enable_deep_supervision
        )
