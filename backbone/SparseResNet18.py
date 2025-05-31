# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from typing import List, Tuple, Dict, Optional
from backbone.utils.k_winners import KWinners2d
from torch.nn.functional import avg_pool2d, relu
import math

def conv3x3(in_planes: int, out_planes: int, stride: int=1) -> nn.Conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SparseBasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int=1, kw_percent_on=0.1, local=False, relu=False, apply_kw=True) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(SparseBasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.apply_kw = apply_kw

        self.kw1 = KWinners2d(
            channels=planes,
            percent_on=kw_percent_on,
            k_inference_factor=1.0,
            boost_strength=0.0,
            boost_strength_factor=0.0,
            local=local,
            relu=relu,
        )

        self.kw2 = KWinners2d(
            channels=planes,
            percent_on=kw_percent_on,
            k_inference_factor=1.0,
            boost_strength=0.0,
            boost_strength_factor=0.0,
            local=local,
            relu=relu,
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """

        out = self.bn1(self.conv1(x))
        if self.apply_kw:
            out = self.kw1(out)
        else:
            out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if self.apply_kw:
            out = self.kw2(out)
        else:
            out = F.relu(out)
        return out


class SparseResNet(nn.Module):
    """
    Sparse ResNet network architecture with k-WTA activations. Designed for complex datasets.
    """

    def __init__(self, block: SparseBasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, kw_percent_on=(0.1, 0.1, 0.1, 0.1), local=False, relu=False, apply_kw=(1, 1, 1, 1)) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(SparseResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.num_blocks = num_blocks

        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1, kw_percent_on=kw_percent_on[0], local=local, relu=relu, apply_kw=apply_kw[0])
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2, kw_percent_on=kw_percent_on[1], local=local, relu=relu, apply_kw=apply_kw[0])
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2, kw_percent_on=kw_percent_on[2], local=local, relu=relu, apply_kw=apply_kw[0])
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2, kw_percent_on=kw_percent_on[3], local=local, relu=relu, apply_kw=apply_kw[0])
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

        self._features = nn.Sequential(self.conv1,
                                       self.bn1,
                                       self.layer1,
                                       self.layer2,
                                       self.layer3,
                                       self.layer4
                                       )

        self.classifier = self.linear

    def _make_layer(self, block: SparseBasicBlock, planes: int,
                    num_blocks: int, stride: int, kw_percent_on: float, local: bool, relu:bool, apply_kw:bool) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, kw_percent_on, local, relu, apply_kw))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_activations=False) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        activations = {}
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  # 64, 32, 32
        out = self.layer2(out)  # 128, 16, 16
        out = self.layer3(out)  # 256, 8, 8
        out = self.layer4(out)  # 512, 4, 4
        out = avg_pool2d(out, out.shape[2]) # 512, 1, 1
        out = out.view(out.size(0), -1)  # 512
        activations['feat'] = out
        out = self.linear(out)
        if return_activations:
            return out, activations
        else:
            return out

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the non-activated output of the second-last layer.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        out = self._features(x)
        out = avg_pool2d(out, out.shape[2])
        feat = out.view(out.size(0), -1)
        return feat

    def get_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the non-activated output of the last convolutional.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        feat = self._features(x)
        out = avg_pool2d(feat, feat.shape[2])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return feat, out

    def extract_features(self, x: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns the non-activated output of the last convolutional.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        out = relu(self.bn1(self.conv1(x)))
        feat1 = self.layer1(out)  # 64, 32, 32
        feat2 = self.layer2(feat1)  # 128, 16, 16
        feat3 = self.layer3(feat2)  # 256, 8, 8
        feat4 = self.layer4(feat3)  # 512, 4, 4
        out = avg_pool2d(feat4, feat4.shape[2])  # 512, 1, 1
        out = out.view(out.size(0), -1)  # 512
        out = self.linear(out)

        return (feat1, feat2, feat3, feat4), out

    def get_features_only(self, x: torch.Tensor, feat_level: int) -> torch.Tensor:
        """
        Returns the non-activated output of the last convolutional.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """

        feat = relu(self.bn1(self.conv1(x)))

        if feat_level > 0:
            feat = self.layer1(feat)  # 64, 32, 32
        if feat_level > 1:
            feat = self.layer2(feat)  # 128, 16, 16
        if feat_level > 2:
            feat = self.layer3(feat)  # 256, 8, 8
        if feat_level > 3:
            feat = self.layer4(feat)  # 512, 4, 4
        return feat

    def predict_from_features(self, feats: torch.Tensor, feat_level: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns the non-activated output of the last convolutional.
        :param feats: input tensor (batch_size, *input_shape)
        :param feat_level: resnet block
        :return: output tensor (??)
        """

        out = feats

        if feat_level < 1:
            out = self.layer1(out)  # 64, 32, 32
        if feat_level < 2:
            out = self.layer2(out)  # 128, 16, 16
        if feat_level < 3:
            out = self.layer3(out)  # 256, 8, 8
        if feat_level < 4:
            out = self.layer4(out)  # 512, 4, 4

        out = avg_pool2d(out, out.shape[2])  # 512, 1, 1
        out = out.view(out.size(0), -1)  # 512
        out = self.linear(out)

        return out

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)


def sparse_resnet18(nclasses: int, nf: int=64, kw_percent_on=(0.9, 0.9, 0.9, 0.9), local=False, relu=False, apply_kw=(1, 1, 1, 1)) -> SparseResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return SparseResNet(SparseBasicBlock, [2, 2, 2, 2], nclasses, nf, kw_percent_on, local, relu, apply_kw)


class LoraLinear(nn.Module):
    is_lora_layer = True  # Cờ để nhận diện tầng LoRA

    def __init__(self, in_features: int, out_features: int, rank: int,
                 num_tasks: int = 10):  # Thêm num_tasks để quản lý
        super().__init__()
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        self.num_total_tasks = num_tasks  # Số lượng tác vụ tối đa dự kiến

        # Tầng linear gốc (đóng băng)
        self.linear = nn.Linear(in_features, out_features)

        # Các tham số LoRA A và B, được lưu trữ trong nn.ModuleDict cho mỗi tác vụ
        # ModuleDict cho phép truy cập bằng key (task_id_str) và tự động đăng ký tham số
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()

        # Scaling factor cho LoRA
        self.scaling = 1.0 / rank  # Một số triển khai dùng alpha / rank, ở đây dùng 1.0 / rank đơn giản

        self._active_task_id: Optional[int] = None

    def initialize_lora_task_adapters(self, task_id: int):
        """Khởi tạo adapter LoRA cho một tác vụ cụ thể."""
        task_id_str = str(task_id)
        if task_id_str not in self.lora_A:
            self.lora_A[task_id_str] = nn.Parameter(torch.zeros(self.rank, self.in_features))
            self.lora_B[task_id_str] = nn.Parameter(torch.zeros(self.out_features, self.rank))
            nn.init.kaiming_uniform_(self.lora_A[task_id_str], a=math.sqrt(5))  # Khởi tạo giống nn.Linear
            # nn.init.zeros_(self.lora_B[task_id_str]) # Thường khởi tạo B là zeros
            print(f"Initialized LoRA adapters for task {task_id} in LoraLinear layer.")

    def set_active_lora_task(self, task_id: Optional[int]):
        """Đặt tác vụ LoRA nào đang hoạt động (để huấn luyện hoặc inference)."""
        self._active_task_id = task_id
        # Đặt requires_grad cho các adapter
        for t_id_str, param_A in self.lora_A.items():
            param_B = self.lora_B[t_id_str]
            is_active_task = (self._active_task_id is not None and t_id_str == str(self._active_task_id))
            param_A.requires_grad = is_active_task
            param_B.requires_grad = is_active_task
        if self._active_task_id is not None and str(self._active_task_id) not in self.lora_A:
            print(
                f"Warning: Active LoRA task {self._active_task_id} set, but adapters not initialized. Call initialize_lora_task_adapters first.")

    def forward(self, x: torch.Tensor,
                lora_weights_to_subtract: Optional[List[Dict[str, torch.Tensor]]] = None) -> torch.Tensor:
        # lora_weights_to_subtract: list các dict, mỗi dict chứa 'A' và 'B' của một tác vụ cũ cần trừ

        # Output từ tầng linear gốc (đã đóng băng)
        base_output = self.linear(x)
        lora_output_combined = torch.zeros_like(base_output)

        if lora_weights_to_subtract is not None:
            # Chế độ "LoRA Subtraction" để tính DRS (W0 - sum(Bj Aj))x
            # Trong trường hợp này, chúng ta đang tính (W0x - sum(Bj Aj x))
            # Điều này hơi khác so với W_bar * x = (W0 - sum(Bj Aj)) * x nếu LoRA được merge trước.
            # Để đơn giản, ta tính W0x và trừ đi các thành phần (Bj Aj x)
            subtracted_lora_effect = torch.zeros_like(base_output)
            for past_lora_w in lora_weights_to_subtract:
                # past_lora_w là dict {'A': tensor, 'B': tensor}
                # Cần đảm bảo key khớp với tên tham số đã lưu
                # Giả định key là 'lora_A_specific_name' và 'lora_B_specific_name'
                # Hoặc đơn giản là A_tensor, B_tensor nếu past_lora_w là tuple (A,B)
                # Ở đây tôi giả định past_lora_w là (A_tensor, B_tensor)
                if isinstance(past_lora_w, tuple) and len(past_lora_w) == 2:
                    lora_A_past, lora_B_past = past_lora_w
                    subtracted_lora_effect += (x @ lora_A_past.T @ lora_B_past.T) * self.scaling
            return base_output - subtracted_lora_effect  # (W0 - sum BA)x

        # Chế độ forward bình thường (huấn luyện hoặc inference)
        # Sử dụng adapter LoRA của tác vụ đang hoạt động (nếu có)
        # Hoặc tổng hợp hiệu ứng của tất cả các LoRA đã học cho inference (nếu _active_task_id is None)
        # Cách tiếp cận LoRA Subtraction DRS thường chỉ huấn luyện adapter của tác vụ hiện tại.
        # Khi inference, nó có thể dùng W_t = W_0 + sum_{j=1 to t} B_j A_j

        active_task_id_str = str(self._active_task_id) if self._active_task_id is not None else None

        if active_task_id_str and active_task_id_str in self.lora_A:
            # Huấn luyện hoặc inference cho một tác vụ cụ thể (chỉ dùng LoRA của tác vụ đó)
            lora_A_active = self.lora_A[active_task_id_str]
            lora_B_active = self.lora_B[active_task_id_str]
            lora_output_combined = (x @ lora_A_active.T @ lora_B_active.T) * self.scaling
            return base_output + lora_output_combined
        elif not active_task_id_str:  # Chế độ inference tổng hợp (ví dụ cuối cùng)
            # Tổng hợp tất cả các LoRA đã học
            for task_str in self.lora_A.keys():
                lora_A_i = self.lora_A[task_str]
                lora_B_i = self.lora_B[task_str]
                lora_output_combined += (x @ lora_A_i.T @ lora_B_i.T) * self.scaling
            return base_output + lora_output_combined
        else:  # Không có LoRA hoạt động hoặc được chỉ định
            return base_output

    def get_task_lora_weights(self, task_id: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Lấy trọng số (A, B) của adapter LoRA cho một tác vụ."""
        task_id_str = str(task_id)
        if task_id_str in self.lora_A:
            return self.lora_A[task_id_str].data.detach().clone(), self.lora_B[task_id_str].data.detach().clone()
        return None

    def get_all_lora_parameters_for_task(self, task_id: int) -> List[nn.Parameter]:
        """Lấy danh sách các tham số LoRA (A và B) cho một tác vụ để đưa vào optimizer."""
        task_id_str = str(task_id)
        params = []
        if task_id_str in self.lora_A:
            params.append(self.lora_A[task_id_str])
            params.append(self.lora_B[task_id_str])
        return params


class SparseResNetLora(nn.Module):  # Đổi tên để phân biệt
    """
    Sparse ResNet với LoRA tích hợp vào tầng Linear cuối cùng.
    """

    def __init__(self, block: type, num_blocks: List[int],  # block là type, không phải instance
                 num_classes: int, nf: int,
                 kw_percent_on=(0.1, 0.1, 0.1, 0.1),
                 local=False, relu_kw=False, apply_kw=(1, 1, 1, 1),  # Đổi tên relu thành relu_kw
                 lora_rank: int = 8,  # Thêm lora_rank
                 num_total_tasks: int = 10  # Số tác vụ dự kiến cho LoRA
                 ):
        super().__init__()
        self.in_planes = nf
        self.block_class = block  # Lưu trữ class của block
        self.num_classes = num_classes
        self.nf = nf
        self.kw_percent_on_layers = kw_percent_on  # kw cho từng stage
        self.local_kw = local
        self.relu_kw = relu_kw
        self.apply_kw_layers = apply_kw  # apply_kw cho từng stage
        self.lora_rank = lora_rank
        self.num_total_tasks = num_total_tasks

        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)

        # Sử dụng giá trị từ tuple/list cho từng layer
        self.layer1 = self._make_layer(nf * 1, num_blocks[0], stride=1, kw_idx=0)
        self.layer2 = self._make_layer(nf * 2, num_blocks[1], stride=2, kw_idx=1)
        self.layer3 = self._make_layer(nf * 4, num_blocks[2], stride=2, kw_idx=2)
        self.layer4 = self._make_layer(nf * 8, num_blocks[3], stride=2, kw_idx=3)

        # Thay thế nn.Linear bằng LoraLinear
        self.lora_linear_classifier = LoraLinear(nf * 8 * self.block_class.expansion, num_classes, rank=self.lora_rank,
                                                 num_tasks=self.num_total_tasks)

        # self._features và self.classifier để tương thích với một số hàm cũ nếu cần
        self._features_extractor = nn.Sequential(self.conv1,
                                                 self.bn1,
                                                 nn.ReLU(),  # Thêm ReLU sau conv1/bn1 đầu tiên
                                                 self.layer1,
                                                 self.layer2,
                                                 self.layer3,
                                                 self.layer4
                                                 )
        self.classifier = self.lora_linear_classifier  # Trỏ classifier vào tầng LoRA

    def _make_layer(self, planes: int, num_blocks_layer: int, stride: int, kw_idx: int) -> nn.Module:
        # Lấy giá trị kw_percent_on và apply_kw cho layer hiện tại
        current_kw_percent = self.kw_percent_on_layers[kw_idx] if isinstance(self.kw_percent_on_layers, (
        list, tuple)) else self.kw_percent_on_layers
        current_apply_kw = self.apply_kw_layers[kw_idx] if isinstance(self.apply_kw_layers,
                                                                      (list, tuple)) else self.apply_kw_layers

        strides = [stride] + [1] * (num_blocks_layer - 1)
        layers = []
        for s_stride in strides:  # Đổi tên biến stride để không xung đột
            layers.append(self.block_class(self.in_planes, planes, s_stride,
                                           kw_percent_on=current_kw_percent,
                                           local=self.local_kw, relu=self.relu_kw,
                                           apply_kw=bool(current_apply_kw)))  # Chuyển apply_kw sang bool
            self.in_planes = planes * self.block_class.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_activations=False,
                active_lora_task_id: Optional[int] = None,  # Cho phép truyền task_id cho LoRA
                lora_weights_to_subtract: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
                # Cho phép truyền weights để trừ
                ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  # Sửa type hint

        # Đặt LoRA task hoạt động cho tầng classifier
        # Nếu bạn có nhiều tầng LoRA, bạn cần set cho tất cả.
        self.lora_linear_classifier.set_active_lora_task(active_lora_task_id)

        activations = {}
        out = relu(self.bn1(self.conv1(x)))  # Thường có ReLU/KWTA sau BN đầu tiên
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, out.shape[2])
        feat_representation = out.view(out.size(0), -1)
        activations['feat'] = feat_representation

        # Forward qua tầng LoraLinear, truyền weights cần trừ nếu có
        if lora_weights_to_subtract is not None and hasattr(self.lora_linear_classifier, 'is_lora_layer'):
            out_classifier = self.lora_linear_classifier(feat_representation,
                                                         lora_weights_to_subtract=lora_weights_to_subtract)
        else:
            out_classifier = self.lora_linear_classifier(feat_representation)

        if return_activations:
            return out_classifier, activations
        else:
            return out_classifier

    # --- Các phương thức quản lý LoRA ---
    def activate_lora_task_adapters(self, task_id: int):
        """Kích hoạt (khởi tạo nếu chưa có) và đặt requires_grad cho adapter LoRA của task_id."""
        # Giả sử chỉ có 1 tầng LoRA là classifier
        self.lora_linear_classifier.initialize_lora_task_adapters(task_id)
        self.lora_linear_classifier.set_active_lora_task(task_id)
        # Nếu có nhiều tầng LoRA, lặp qua chúng và gọi các hàm tương ứng

    def get_task_lora_weights(self, task_id: int) -> Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """Lấy trọng số LoRA (A,B) cho một tác vụ từ tất cả các tầng LoRA."""
        weights = {}
        # Giả sử chỉ có classifier là LoRA
        lora_classifier_weights = self.lora_linear_classifier.get_task_lora_weights(task_id)
        if lora_classifier_weights:
            # Key có thể là tên của module để dễ dàng map lại
            weights['lora_linear_classifier'] = lora_classifier_weights
        return weights if weights else None

    def get_lora_module_name_for_param(self, param_name: str) -> Optional[str]:
        """Map tên tham số LoRA về tên module LoRA.
           Ví dụ: 'classifier.lora_A.0' -> 'classifier' (nếu classifier là LoraLinear)
        """
        # Logic này cần được điều chỉnh tùy theo cách bạn đặt tên các tầng LoRA
        # và cách các tham số LoRA được lưu trữ (ví dụ trong ModuleDict của LoraLinear)
        if 'lora_linear_classifier.lora_A' in param_name or 'lora_linear_classifier.lora_B' in param_name:
            return 'lora_linear_classifier'  # Đây là key bạn sẽ dùng cho drs_projection_matrices
        # Thêm các trường hợp khác nếu có nhiều tầng LoRA
        return None

    def get_state_with_past_loras_subtracted(self, past_lora_weights_list: List[
        Dict[str, Tuple[torch.Tensor, torch.Tensor]]]):
        """
        Trả về một state_dict hoặc một model mới mà trọng số của nó tương ứng với W_0 - sum(past_loras).
        Hàm này phức tạp vì LoRA không trực tiếp sửa W_0.
        Thay vào đó, khi forward, chúng ta sẽ tính (W_0 * x) - sum(B_j * A_j * x).
        Vì vậy, hàm này có thể không trả về state_dict mà chỉ là cờ/thông tin cho hàm forward.
        Hoặc, nó có thể chuẩn bị một danh sách các (A,B) cần trừ cho mỗi tầng LoRA.
        """
        # Cách đơn giản là hàm forward của LoraLinear sẽ nhận danh sách các (A,B) cần trừ.
        # Hàm này chỉ cần chuẩn bị danh sách đó.
        # past_lora_weights_list: [{'lora_linear_classifier': (A_tensor, B_tensor)}, ...]

        # Giả sử chúng ta chỉ có một tầng LoRA là classifier
        # Danh sách các (A,B) của tầng classifier từ các tác vụ cũ
        classifier_loras_to_subtract = []
        for task_weights_dict in past_lora_weights_list:
            if 'lora_linear_classifier' in task_weights_dict:
                classifier_loras_to_subtract.append(task_weights_dict['lora_linear_classifier'])

        # Trả về một dict mà hàm forward có thể sử dụng
        return {'lora_linear_classifier': classifier_loras_to_subtract}

    # Giữ lại các hàm features, get_features, ... nếu chúng vẫn cần thiết và tương thích
    # Có thể cần điều chỉnh chúng để hoạt động với LoraLinear.
    # Ví dụ, hàm features:
    def features(self, x: torch.Tensor) -> torch.Tensor:
        out = self._features_extractor(x)
        out = avg_pool2d(out, out.shape[2])
        feat = out.view(out.size(0), -1)
        return feat

    # Các hàm get_params, set_params, get_grads có thể cần xem xét lại
    # vì giờ đây một phần model (PTM) bị đóng băng và LoRA adapters được thêm/bớt động.
    # get_params nên chỉ trả về các tham số có thể huấn luyện (trainable parameters).


# Hàm khởi tạo sparse_resnet18_lora
def sparse_resnet18_lora(nclasses: int, nf: int = 64,
                         kw_percent_on=(0.9, 0.9, 0.9, 0.9),
                         local=False, relu_kw=False, apply_kw=(1, 1, 1, 1),
                         lora_rank: int = 8, num_total_tasks: int = 5
                         ) -> SparseResNetLora:
    # apply_kw cần là một tuple/list có 4 phần tử boolean hoặc int 0/1
    # Đảm bảo apply_kw có đúng 4 phần tử, nếu không thì dùng giá trị đầu cho tất cả
    if not isinstance(apply_kw, (list, tuple)) or len(apply_kw) != 4:
        apply_kw_parsed = [bool(apply_kw[0]) if isinstance(apply_kw, (list, tuple)) else bool(apply_kw)] * 4
    else:
        apply_kw_parsed = [bool(val) for val in apply_kw]

    return SparseResNetLora(SparseBasicBlock, [2, 2, 2, 2], nclasses, nf,
                            kw_percent_on, local, relu_kw, apply_kw_parsed,  # Truyền apply_kw đã parse
                            lora_rank, num_total_tasks)