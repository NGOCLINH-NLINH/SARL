import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from copy import deepcopy
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from backbone.MNISTMLP import SparseMNISTMLP
from backbone.SparseResNet18 import sparse_resnet18
from models.utils.losses import SupConLoss
from models.utils.pos_groups import class_dict, pos_groups
from typing import List, Tuple, Dict
from datasets.utils.continual_dataset import ContinualDataset


num_classes_dict = {
    'seq-cifar10': 10,
    'seq-cifar100': 100,
    'seq-tinyimg': 200,
    'gcil-cifar100': 100,
    'seq-imagenet100': 100,
}


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='SARL with LoRA Subtraction DRS')
    # --- GIẢ ĐỊNH add_management_args, add_experiment_args, add_rehearsal_args tồn tại ---
    # add_management_args(parser)
    # add_experiment_args(parser)
    # add_rehearsal_args(parser)
    # Các args cơ bản để chạy được
    parser.add_argument('--experiment_id', type=str, default='test_sarl_drs')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dataset', type=str, default='seq-cifar100')
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--n_epochs', type=int, default=50) # Đã có trong script
    parser.add_argument('--buffer_size', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32) # Đã có trong script
    parser.add_argument('--minibatch_size', type=int, default=32) # size của batch lấy từ buffer
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--csv_log', action='store_true')
    parser.add_argument('--notes', type=str, default=None)
    parser.add_argument('--num_tasks', type=int, default=10, help="Total number of tasks, used for LoRA init")


    # Consistency Regularization Weight (SARL)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1.0) # Đảm bảo có giá trị default
    parser.add_argument('--op_weight', type=float, default=0.1)
    parser.add_argument('--sim_thresh', type=float, default=0.80)
    parser.add_argument('--sm_weight', type=float, default=0.01)
    # Sparsity param (SARL)
    parser.add_argument('--apply_kw', nargs='*', type=int, default=[1, 1, 1, 1])
    # parser.add_argument('--kw', type=float, nargs='*', default=[0.9, 0.9, 0.9, 0.9]) # Chuyển thành string để parse
    parser.add_argument('--kw', type=str, default='0.9 0.9 0.9 0.9', help='k-WTA percentages as space-separated string')

    parser.add_argument('--kw_relu', type=int, default=1)
    parser.add_argument('--kw_local', type=int, default=1)
    parser.add_argument('--num_feats', type=int, default=512)
    # Experimental Args (SARL)
    parser.add_argument('--save_interim', type=int, default=0) # Đổi default từ script
    parser.add_argument('--warmup_epochs', type=int, default=3) # Đổi default từ script
    parser.add_argument('--use_lr_scheduler', type=int, default=1)
    # parser.add_argument('--lr_steps', type=int, nargs='*', default=[35, 45]) # Chuyển thành string
    parser.add_argument('--lr_steps', type=str, default='35 45', help='LR decay steps as space-separated string')


    # LoRA DRS Args (MỚI)
    parser.add_argument('--lora_rank', type=int, default=8, help="Rank for LoRA adapters")
    parser.add_argument('--drs_variance_threshold', type=float, default=0.95, help="Cumulative variance for DRS principal components")
    parser.add_argument('--lambda_atl', type=float, default=0.1, help="Weight for Augmented Triplet Loss (ATL)")
    parser.add_argument('--atl_margin', type=float, default=0.2, help="Margin for ATL")
    parser.add_argument('--max_samples_for_drs', type=int, default=512, help="Max samples to compute DRS projection matrix")

    return parser

class SARLDRS(ContinualModel):
    NAME = 'sarl_drs'
    COMPATIBILITY = ['class-il']

    def __init__(self, backbone_class_fn, loss_fn, args, transform):
        # backbone_class_fn là một hàm trả về instance của backbone (ví dụ lambda: sparse_resnet18_lora(...))
        # Điều này cho phép truyền các args cụ thể vào backbone từ đây.
        super().__init__(None, loss_fn, args, transform)  # Truyền None cho backbone

        if self.args.buffer_size > 0:
            self.buffer = Buffer(self.args.buffer_size, self.device)
        else:
            self.buffer = None  # Hỗ trợ chạy exemplar-free

        # Chuyển đổi kw và lr_steps từ string (nếu cần)
        if isinstance(self.args.kw, str):
            self.args.kw = [float(x) for x in self.args.kw.split()]
        if isinstance(self.args.lr_steps, str):
            self.args.lr_steps = [int(x) for x in self.args.lr_steps.split()]

        # --- THAY ĐỔI KHỞI TẠO BACKBONE ---
        if 'mnist' in self.args.dataset:
            # self.net = SparseMNISTMLP(...) # Cần phiên bản LoRA cho MNIST MLP nếu muốn
            raise NotImplementedError("LoRA version for MNIST MLP not implemented in this example.")
        else:
            # Gọi hàm lambda đã truyền vào để tạo backbone với các args cần thiết
            self.net = backbone_class_fn(
                nclasses=num_classes_dict[args.dataset],
                kw_percent_on=self.args.kw,  # Đã parse
                local=bool(self.args.kw_local),
                relu_kw=bool(self.args.kw_relu),  # Đổi tên từ relu
                apply_kw=[bool(x) for x in self.args.apply_kw],  # Đảm bảo là bool
                lora_rank=self.args.lora_rank,
                num_total_tasks=self.args.num_tasks  # Giả sử có args.num_tasks
            ).to(self.device)

        # Đóng băng PTM gốc, chỉ các LoRA adapters là trainable
        for name, param in self.net.named_parameters():
            if 'lora_' not in name.lower():  # Kiểm tra tổng quát hơn
                param.requires_grad = False
            else:  # Các LoRA adapters ban đầu cũng đóng băng, sẽ được kích hoạt theo từng tác vụ
                param.requires_grad = False

        self.net_old = None
        # Optimizer sẽ được tạo trong begin_task

        # --- CÁC THUỘC TÍNH MỚI CHO LoRA VÀ DRS ---
        self.past_lora_task_weights: List[
            Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = []  # List các dict [{lora_module_name: (A, B)}, ...]
        self.drs_projection_matrices: Dict[str, torch.Tensor] = {}  # {lora_module_name: P_tl}

        # --- CÁC THUỘC TÍNH CỦA SARL GỐC ---
        self.alpha = args.alpha
        self.beta = args.beta  # Thêm beta vào đây
        self.op_weight = args.op_weight
        self.sim_thresh = args.sim_thresh
        self.sm_weight = args.sm_weight
        self.lambda_atl = args.lambda_atl  # Mới
        self.atl_margin = args.atl_margin  # Mới

        self.current_task = 0
        self.epoch = 0  # Epoch hiện tại trong một tác vụ
        # self.global_step = 0 # Nếu cần
        # self.lst_models = ['net'] # Nếu dùng cho việc lưu trữ

        self.op = torch.zeros(num_classes_dict[args.dataset], args.num_feats, device=self.device)
        self.op_sum = torch.zeros(num_classes_dict[args.dataset], args.num_feats, device=self.device)
        self.sample_counts = torch.zeros(num_classes_dict[args.dataset], device=self.device)
        self.running_op = torch.zeros(num_classes_dict[args.dataset], args.num_feats, device=self.device)
        self.running_sample_counts = torch.zeros(num_classes_dict[args.dataset], device=self.device)
        self.learned_classes = []
        # self.flag = True # Có thể không cần nữa
        self.eval_prototypes_for_lsm = True  # Cờ riêng cho việc tính pos_groups
        self.pos_groups = {}
        self.dist_mat = torch.zeros(num_classes_dict[args.dataset], num_classes_dict[args.dataset], device=self.device)
        self.class_dict_internal = class_dict.get(args.dataset, {i:str(i) for i in range(num_classes_dict[args.dataset])} )

        # --- PHƯƠNG THỨC MỚI CHO QUẢN LÝ TÁC VỤ VÀ DRS ---
        def begin_task(self, dataset: ContinualDataset) -> None:  # dataset là ContinualDataset object
            """Được gọi ở đầu mỗi tác vụ."""
            self.epoch = 0  # Reset epoch counter
            self.eval_prototypes_for_lsm = True  # Sẵn sàng tính pos_groups sau warmup

            # Kích hoạt adapter LoRA cho tác vụ hiện tại
            # Hàm này trong backbone_lora sẽ tạo params A, B mới và đặt requires_grad=True
            self.net.activate_lora_task_adapters(self.current_task)

            self.get_optimizer()  # Tạo optimizer chỉ cho các params LoRA mới

            if self.current_task > 0:
                # train_loader của dataset object cho tác vụ hiện tại
                self.compute_drs_projection_matrices(dataset.train_loader)

            if self.current_task == 0:  # Reset cho tác vụ đầu tiên
                self.net_old = None
                self.drs_projection_matrices = {}
                self.past_lora_task_weights = []

        def get_optimizer(self):
            """Chỉ tối ưu hóa các tham số LoRA của tác vụ hiện tại (có requires_grad=True)."""
            current_trainable_params = [p for p in self.net.parameters() if p.requires_grad]

            if not current_trainable_params:
                print(f"Warning: No trainable LoRA parameters found for task {self.current_task} to optimize.")
                self.opt = None
                self.scheduler = None
                return

            # Bạn có thể thêm lựa chọn optimizer từ args nếu muốn
            self.opt = SGD(current_trainable_params, lr=self.args.lr, momentum=0.9, weight_decay=5e-4)

            if self.args.use_lr_scheduler:
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, self.args.lr_steps, gamma=0.1)
            else:
                self.scheduler = None

        def _get_lora_layer_input_features(self, train_loader_curr_task) -> Dict[str, List[torch.Tensor]]:
            """Helper để thu thập features X_t^l cho việc tính DRS."""
            self.net.eval()  # Quan trọng: đặt eval mode cho network chính

            # Tạo temp_net bằng cách áp dụng LoRA Subtraction
            temp_net_state_dict = self.net.get_state_dict_with_past_loras_subtracted(self.past_lora_task_weights)

            # Tạo một instance mới của backbone cho temp_net để tránh thay đổi self.net
            # Điều này đảm bảo self.net không bị ảnh hưởng bởi việc load state_dict tạm thời.
            # Tuy nhiên, việc này tốn bộ nhớ. Một cách khác là deepcopy(self.net) rồi load.
            # Cách tối ưu hơn là hàm get_state_dict_with_past_loras_subtracted sẽ trả về một model "ảo"
            # chỉ dùng để forward một lần.
            # Ở đây, giả sử backbone có hàm load_state_dict_for_lora_subtraction
            temp_net = deepcopy(self.net)  # Cách an toàn nhưng có thể chậm/tốn bộ nhớ
            temp_net.load_state_dict(temp_net_state_dict)
            temp_net.eval()

            layer_inputs_for_drs: Dict[str, List[torch.Tensor]] = {}
            hooks = []

            for name, module in temp_net.named_modules():
                if hasattr(module, 'is_lora_layer') and module.is_lora_layer:
                    def hook_fn(mod, inp, outp, key=name):  # Dùng key để tránh lỗi closure
                        if key not in layer_inputs_for_drs: layer_inputs_for_drs[key] = []
                        # inp[0] là input tensor cho nn.Linear
                        # Cần detach().cpu() để giải phóng GPU memory nếu features lớn
                        layer_inputs_for_drs[key].append(
                            inp[0].detach())  # Giữ trên device hiện tại để tính toán nhanh hơn

                    hooks.append(module.register_forward_hook(hook_fn))

            num_samples_collected = 0
            with torch.no_grad():
                for inputs, _, _ in train_loader_curr_task:
                    inputs = inputs.to(self.device)
                    # Khi gọi forward trên temp_net, nó KHÔNG nên sử dụng các active_lora_task_id
                    # vì chúng ta muốn W_bar * x
                    _ = temp_net(inputs, active_lora_task_id=None)  # Không active LoRA nào khi tính X_t^l
                    num_samples_collected += inputs.size(0)
                    if num_samples_collected >= self.args.max_samples_for_drs:
                        break

            for h in hooks: h.remove()
            # Không cần self.net.train() ở đây vì self.net không bị thay đổi
            return layer_inputs_for_drs

        def compute_drs_projection_matrices(self, train_loader_curr_task: DataLoader):
            print(f"Task {self.current_task}: Computing DRS projection matrices...")
            layer_inputs_for_drs = self._get_lora_layer_input_features(train_loader_curr_task)

            self.drs_projection_matrices = {}  # Reset cho tác vụ hiện tại
            for lora_module_name, features_list in layer_inputs_for_drs.items():
                if not features_list: continue

                X_tilde_l_tensor = torch.cat(features_list, dim=0)  # Đã ở trên device
                if X_tilde_l_tensor.ndim > 2:
                    X_tilde_l_tensor = X_tilde_l_tensor.view(X_tilde_l_tensor.shape[0], -1)
                if X_tilde_l_tensor.shape[0] == 0 or X_tilde_l_tensor.shape[1] == 0: continue

                n_t = X_tilde_l_tensor.shape[0]
                # Eq. 7: X_bar_l = (1/n_t) * (X_tilde_l)^T * X_tilde_l (shape: d_in x d_in)
                # Chuyển sang float32 để SVD ổn định hơn nếu đang dùng half precision
                X_bar_l = (1.0 / n_t) * (X_tilde_l_tensor.float().T @ X_tilde_l_tensor.float())

                try:  # Eq. 10: SVD
                    U, S_singular_values, _ = torch.linalg.svd(X_bar_l, full_matrices=False)
                except Exception as e:
                    print(
                        f"SVD failed for LoRA module {lora_module_name} on device {X_bar_l.device}: {e}. Skipping DRS.")
                    continue

                if torch.any(torch.isnan(S_singular_values)):
                    print(f"NaN singular values for LoRA module {lora_module_name}. Skipping DRS.")
                    continue

                sum_eigenvalues = torch.sum(S_singular_values)
                if sum_eigenvalues < 1e-9:  # Kiểm tra giá trị rất nhỏ
                    print(f"Sum of singular values is near zero for {lora_module_name}. Skipping DRS.")
                    continue

                cumulative_variance = torch.cumsum(S_singular_values, dim=0) / sum_eigenvalues
                k_indices = torch.where(cumulative_variance >= self.args.drs_variance_threshold)[0]

                k = U.shape[1]  # Mặc định lấy tất cả
                if len(k_indices) > 0: k = k_indices[0].item() + 1

                P_tl = U[:, :k]  # Eq. 11 (shape: d_in x k)
                self.drs_projection_matrices[lora_module_name] = P_tl
                print(
                    f"  LoRA module {lora_module_name}: DRS matrix P_t^l shape {P_tl.shape} (d_in={U.shape[1]}, k={k})")

        def _project_gradients(self):
            """
            Chiếu gradient của các tham số LoRA hiện tại (A_t, B_t) vào DRS.
            Đây là phần phức tạp và cần nghiên cứu kỹ Eq. 9 từ bài báo LoRA Subtraction.
            Eq. 9: delta_w_ts^l = P_t^l (P_t^l)^T g_ts^l
            g_ts^l là gradient của *toàn bộ* params của layer l. Với LoRA, ta chỉ có grad của A_t, B_t.
            Cách tiếp cận khả thi:
            1. Vector hóa grad(A_t) và grad(B_t).
            2. Nếu P_t^l (từ không gian input X_t^l) có thể dùng để chiếu grad(A_t) (vì A_t hoạt động trên X_t^l).
            3. Cần một P'_t^l (từ không gian output Z_t^l) để chiếu grad(B_t). Bài báo LoRA Subtraction không nói rõ điều này.
            TẠM THỜI BỎ QUA PHÉP CHIẾU THỰC SỰ ĐỂ MÃ CHẠY ĐƯỢC.
            """
            if not self.drs_projection_matrices or self.current_task == 0: return

            with torch.no_grad():
                for name, param in self.net.named_parameters():
                    if param.grad is not None and param.requires_grad:  # Chỉ LoRA của task hiện tại
                        # lora_module_name = self.net.get_lora_module_name_for_param(name, self.current_task) # Cần hàm này
                        # if lora_module_name and lora_module_name in self.drs_projection_matrices:
                        #     P_tl = self.drs_projection_matrices[lora_module_name]
                        #     # Logic chiếu phức tạp ở đây...
                        #     # Ví dụ cho grad(A_t) (shape: rank, d_in): chiếu từng hàng (vector d_in)
                        #     if 'lora_A' in name: # Giả định grad(A) có thể chiếu bằng P_tl từ X_t^l
                        #         grad_A_rows = param.grad.data # (rank, d_in)
                        #         projected_grad_A_rows = grad_A_rows @ P_tl @ P_tl.T
                        #         param.grad.data = projected_grad_A_rows
                        #     # Tương tự cho grad(B_t) nếu có P'_t^l
                        # else:
                        # print(f"Skipping grad projection for {name} - no DRS matrix or complex application.")
                        pass  # BỎ QUA ĐỂ TEST

        def observe(self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs: torch.Tensor) -> float:
            if self.opt is None:  # Trường hợp không có gì để tối ưu
                if self.buffer is not None:
                    with torch.no_grad():
                        outputs_no_opt, _ = self.net(inputs, return_activations=True,
                                                     active_lora_task_id=self.current_task)
                    self.buffer.add_data(
                        examples=not_aug_inputs.cpu(),
                        labels=labels.cpu(),
                        logits=outputs_no_opt.data.detach().cpu()
                    )
                return 0.0

            real_batch_size = inputs.shape[0]
            self.opt.zero_grad()
            self.net.train()  # Đảm bảo model ở train mode
            total_loss_value = torch.tensor(0.0, device=self.device)

            # Truyền active_lora_task_id vào forward của self.net
            active_lora_id = self.current_task

            # 1. Loss từ Buffer (SARL)
            if self.buffer is not None and not self.buffer.is_empty():
                buf_inputs, buf_labels, buf_logits_target = self.buffer.get_data(self.args.minibatch_size,
                                                                                 transform=self.transform)
                buf_inputs, buf_labels, buf_logits_target = buf_inputs.to(self.device), buf_labels.to(
                    self.device), buf_logits_target.to(self.device)

                # Khi forward cho buffer, LoRA của tác vụ hiện tại (active_lora_id) vẫn hoạt động
                # để tính L_OP và L_FR một cách nhất quán với cách nó sẽ thấy dữ liệu cũ.
                # Hoặc, có thể muốn forward với tổng hợp các LoRA cũ cho buffer (nếu net_old dùng cách đó).
                # Giữ nguyên dùng active_lora_id cho đơn giản và nhất quán với cách SARL gốc hoạt động.
                buf_outputs, buf_activations = self.net(buf_inputs, return_activations=True,
                                                        active_lora_task_id=active_lora_id)
                buf_feats = buf_activations['feat']

                loss_buf_ce = self.loss_fn(buf_outputs, buf_labels)
                total_loss_value += loss_buf_ce

                loss_buf_reg = self.args.alpha * F.mse_loss(buf_outputs, buf_logits_target)  # L_FR part 1
                total_loss_value += loss_buf_reg

                if self.current_task > 0 and self.op_weight > 0:  # L_OP
                    buf_feats_norm = F.normalize(buf_feats, p=2, dim=1)
                    loss_op_val = torch.tensor(0.0, device=self.device)
                    unique_buf_labels = torch.unique(buf_labels)
                    num_op_terms = 0
                    for class_idx_tensor in unique_buf_labels:
                        class_idx = class_idx_tensor.item()  # Chuyển sang Python int
                        if class_idx in self.learned_classes:
                            mask = (buf_labels == class_idx_tensor)
                            if mask.sum() > 0:
                                mean_feat_buf = buf_feats_norm[mask].mean(dim=0)
                                # Đảm bảo self.op[class_idx] không phải là zero vector nếu lớp đó đã học
                                if torch.norm(self.op[class_idx]) > 1e-6:
                                    loss_op_val += F.mse_loss(mean_feat_buf, self.op[class_idx])
                                    num_op_terms += 1
                    if num_op_terms > 0: total_loss_value += self.op_weight * (loss_op_val / num_op_terms)

            # 2. Loss từ Dữ liệu Hiện Tại
            outputs_curr, activations_curr = self.net(inputs, return_activations=True,
                                                      active_lora_task_id=active_lora_id)
            feats_curr = activations_curr['feat']

            loss_curr_ce = self.loss_fn(outputs_curr, labels)
            total_loss_value += loss_curr_ce

            # L_FR (SARL) cho dữ liệu mới
            if self.epoch >= self.args.warmup_epochs and self.current_task > 0 and self.net_old is not None:
                with torch.no_grad():
                    # net_old đã bao gồm các LoRA cũ, nên không cần truyền active_lora_task_id cụ thể
                    # hoặc truyền None để nó tổng hợp các LoRA đã có trong nó.
                    outputs_old_target = self.net_old(inputs, active_lora_task_id=None)
                loss_curr_reg = self.args.beta * F.mse_loss(outputs_curr, outputs_old_target)
                total_loss_value += loss_curr_reg

            # L_SM (SARL) cho dữ liệu mới
            if self.epoch >= self.args.warmup_epochs and self.current_task > 0 and self.sm_weight > 0 and self.pos_groups:
                feats_curr_norm_sm = F.normalize(feats_curr, p=2, dim=1)
                loss_sm_val = torch.tensor(0.0, device=self.device)
                num_sm_terms = 0

                batch_prototypes_sm = {}
                unique_labels_curr_batch = torch.unique(labels)
                new_labels_in_batch_sm = [l.item() for l in unique_labels_curr_batch if
                                          l.item() not in self.learned_classes]

                for l_idx_item in new_labels_in_batch_sm:
                    mask = (labels == l_idx_item)
                    if mask.sum() > 0: batch_prototypes_sm[l_idx_item] = feats_curr_norm_sm[mask].mean(dim=0)

                for new_class_idx_item in new_labels_in_batch_sm:
                    if new_class_idx_item not in batch_prototypes_sm: continue
                    anchor_proto_sm = batch_prototypes_sm[new_class_idx_item]
                    current_pos_dist_sm, current_neg_dist_sm = 0.0, 0.0
                    num_pos_pairs, num_neg_pairs = 0, 0

                    all_compare_labels_sm = self.learned_classes + [l for l in new_labels_in_batch_sm if
                                                                    l != new_class_idx_item]

                    for ref_class_idx_item in all_compare_labels_sm:
                        ref_proto_sm = None
                        if ref_class_idx_item in self.learned_classes:
                            ref_proto_sm = self.op[ref_class_idx_item]
                        elif ref_class_idx_item in batch_prototypes_sm:
                            ref_proto_sm = batch_prototypes_sm[ref_class_idx_item]

                        if ref_proto_sm is None or torch.norm(ref_proto_sm) < 1e-6: continue

                        dist_sq = F.mse_loss(anchor_proto_sm, ref_proto_sm)
                        if ref_class_idx_item in self.pos_groups.get(new_class_idx_item, []):
                            current_pos_dist_sm += dist_sq;
                            num_pos_pairs += 1
                        else:
                            current_neg_dist_sm += dist_sq;
                            num_neg_pairs += 1

                    if num_pos_pairs > 0 and num_neg_pairs > 0 and current_neg_dist_sm > 1e-6:
                        loss_sm_val += (current_pos_dist_sm / num_pos_pairs) / (current_neg_dist_sm / num_neg_pairs)
                        num_sm_terms += 1

                if num_sm_terms > 0: total_loss_value += self.sm_weight * (loss_sm_val / num_sm_terms)

            # Augmented Triplet Loss (ATL)
            if self.lambda_atl > 0 and self.current_task > 0:
                atl_loss_terms = []
                anchor_features_norm_atl = F.normalize(feats_curr, p=2, dim=1)
                for i in range(real_batch_size):
                    anchor_feat_atl = anchor_features_norm_atl[i]
                    anchor_label_atl = labels[i].item()

                    positive_mask_atl = (labels == labels[i]) & (torch.arange(real_batch_size, device=self.device) != i)
                    e_ap_atl = torch.tensor(0.0, device=self.device)
                    if positive_mask_atl.sum() > 0:
                        e_ap_atl = F.pairwise_distance(anchor_feat_atl.unsqueeze(0),
                                                       anchor_features_norm_atl[positive_mask_atl]).max()

                    negative_mask_batch_atl = (labels != labels[i])
                    e_an_batch_atl = torch.tensor(float('inf'), device=self.device)
                    if negative_mask_batch_atl.sum() > 0:
                        e_an_batch_atl = F.pairwise_distance(anchor_feat_atl.unsqueeze(0),
                                                             anchor_features_norm_atl[negative_mask_batch_atl]).min()

                    e_an_proto_atl = torch.tensor(float('inf'), device=self.device)
                    old_proto_indices_atl = [l for l in self.learned_classes if
                                             l != anchor_label_atl and torch.norm(self.op[l]) > 1e-6]
                    if old_proto_indices_atl:
                        e_an_proto_atl = F.pairwise_distance(anchor_feat_atl.unsqueeze(0),
                                                             self.op[old_proto_indices_atl]).min()

                    e_an_atl = torch.min(e_an_batch_atl, e_an_proto_atl)
                    if not torch.isinf(e_an_atl) and e_an_atl < float('inf'):  # Đảm bảo có negative hợp lệ
                        atl_loss_terms.append(F.relu(e_ap_atl - e_an_atl + self.atl_margin))

                if atl_loss_terms: total_loss_value += self.lambda_atl * (torch.stack(atl_loss_terms).mean())

            if torch.isnan(total_loss_value): raise ValueError('SARL_DRS: NaN Loss observed')

            total_loss_value.backward()
            self._project_gradients()  # TẠM THỜI KHÔNG LÀM GÌ BÊN TRONG
            if self.opt: self.opt.step()

            if self.buffer is not None:
                self.buffer.add_data(
                    examples=not_aug_inputs.cpu(),
                    labels=labels[:real_batch_size].cpu(),
                    logits=outputs_curr.data.detach().cpu()
                )
            return total_loss_value.item()

        # dataset là ContinualDataset object, có thuộc tính train_loader cho tác vụ hiện tại
        def end_epoch(self, dataset: ContinualDataset, current_epoch_num: int) -> None:
            self.epoch = current_epoch_num
            if self.scheduler is not None: self.scheduler.step()

            self.net.eval()  # Đặt model ở eval mode cho việc tính prototype

            # Tính self.pos_groups cho L_SM của các epoch tiếp theo trong tác vụ này
            if self.epoch >= self.args.warmup_epochs and self.eval_prototypes_for_lsm and self.current_task > 0:
                print(f'[Task {self.current_task}/Epoch {self.epoch}] Evaluating intermediate prototypes for L_SM...')

                # Reset running_op và counts cho mỗi lần tính toán lại
                self.running_op.zero_()
                self.running_sample_counts.zero_()

                # Thu thập features từ train_loader của tác vụ hiện tại
                # dataset.train_loader ở đây là loader cho (inputs, labels, not_aug_inputs)
                with torch.no_grad():
                    for inputs, labels, _ in dataset.train_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        # Sử dụng LoRA của tác vụ hiện tại để trích xuất feature
                        _, activations = self.net(inputs, return_activations=True,
                                                  active_lora_task_id=self.current_task)
                        feat_norm = F.normalize(activations['feat'], p=2, dim=1)

                        for class_label_tensor in torch.unique(labels):
                            class_label_idx = class_label_tensor.long()  # Chuyển sang Long để làm index
                            mask = (labels == class_label_tensor)
                            self.running_op[class_label_idx] += feat_norm[mask].sum(dim=0)  # Đã ở trên device
                            self.running_sample_counts[class_label_idx] += mask.sum()

                # Tính trung bình cho running_op
                unique_labels_in_task = [idx for idx, count in enumerate(self.running_sample_counts) if count > 0]
                for class_label_idx in unique_labels_in_task:
                    self.running_op[class_label_idx] /= self.running_sample_counts[class_label_idx]

                # Tính self.dist_mat và self.pos_groups (logic tương tự SARL gốc)
                new_labels_task = [l for l in unique_labels_in_task if l not in self.learned_classes]
                all_relevant_labels = self.learned_classes + new_labels_task
                cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)

                current_dist_mat = torch.zeros_like(self.dist_mat)  # Ma trận tạm thời

                for new_l_idx in new_labels_task:
                    proto_new = self.running_op[new_l_idx]
                    if torch.norm(proto_new) < 1e-6: continue
                    for ref_l_idx in all_relevant_labels:
                        if ref_l_idx in new_labels_task:  # New vs New
                            proto_ref = self.running_op[ref_l_idx]
                        else:  # New vs Old (ref_l_idx in self.learned_classes)
                            proto_ref = self.op[ref_l_idx]
                        if torch.norm(proto_ref) < 1e-6: continue
                        current_dist_mat[new_l_idx, ref_l_idx] = cos_sim(proto_new, proto_ref)

                self.pos_groups = {}  # Reset
                print('*' * 30)
                print(f'Positive Groups for L_SM (Task {self.current_task}):')
                for new_l_idx in new_labels_task:
                    sim_values = current_dist_mat[new_l_idx, all_relevant_labels]
                    positive_mask = sim_values > self.args.sim_thresh
                    # Loại trừ chính nó ra khỏi positive group
                    self.pos_groups[new_l_idx] = [label_item for i, label_item in enumerate(all_relevant_labels) if
                                                  positive_mask[i].item() and label_item != new_l_idx]

                    # In ra (tương tự SARL gốc)
                    # class_names_pos_group = [self.class_dict_internal.get(idx, str(idx)) for idx in self.pos_groups[new_l_idx]]
                    # print(f'  {self.class_dict_internal.get(new_l_idx, str(new_l_idx))}: {", ".join(class_names_pos_group)}')

                print('*' * 30)
                self.eval_prototypes_for_lsm = False  # Đánh dấu đã tính cho các epoch sau của tác vụ này

        def end_task(self, dataset: ContinualDataset) -> None:  # dataset là của tác vụ vừa hoàn thành
            self.net.eval()

            # 1. Lưu trữ trọng số LoRA đã học của tác vụ hiện tại
            # Hàm này trong backbone_lora trả về dict {lora_module_name: (A_tensor, B_tensor)}
            current_task_lora_module_weights = self.net.get_task_lora_module_weights(self.current_task)
            if current_task_lora_module_weights:
                self.past_lora_task_weights.append(deepcopy(current_task_lora_module_weights))

            # 2. Cập nhật self.net_old
            self.net_old = deepcopy(self.net)
            self.net_old.eval()

            # 3. Điều chỉnh Batch Norm qua buffer (nếu có buffer)
            if self.buffer is not None and not self.buffer.is_empty():
                all_buf_inputs, _, _ = self.buffer.get_all_data(transform=None)  # Lấy raw data
                if all_buf_inputs.shape[0] > 0:
                    self.net.train()  # BN update ở train mode
                    with torch.no_grad():
                        for i in range(0, all_buf_inputs.shape[0],
                                       self.args.batch_size):  # Dùng batch_size thay vì minibatch_size
                            batch_buf = all_buf_inputs[i:i + self.args.batch_size].to(self.device)
                            # Áp dụng transform nếu buffer lưu raw data
                            # Giả sử buffer đã lưu data đã transform, hoặc self.transform xử lý được
                            if self.transform is not None:
                                # Cần cẩn thận nếu transform yêu cầu PIL Image
                                # Nếu buffer lưu tensor, transform phải chấp nhận tensor
                                # Hoặc, get_all_data trả về PIL Image nếu transform=None
                                pass  # Tạm bỏ qua transform phức tạp ở đây
                            _ = self.net(batch_buf, active_lora_task_id=None)  # Dùng tất cả LoRA đã học cho BN
                    self.net.eval()

            # 4. Finalize Object Prototypes (self.op) cho các lớp trong tác vụ vừa học
            print(f'[Task {self.current_task}] Finalizing Object Prototypes...')
            self.net.eval()
            # Reset op_sum và sample_counts cho các lớp SẼ được cập nhật từ tác vụ này
            # Hoặc nếu là cộng dồn thì không cần reset hoàn toàn.
            # Mã SARL gốc dường như cộng dồn vào op_sum/sample_counts toàn cục.
            # Chúng ta sẽ tính toán lại op_sum và sample_counts chỉ từ dữ liệu của tác vụ hiện tại
            # và sau đó cập nhật self.op.

            # Reset các biến tạm thời
            current_task_op_sum = torch.zeros_like(self.op_sum)
            current_task_sample_counts = torch.zeros_like(self.sample_counts)

            with torch.no_grad():
                for inputs, labels, _ in dataset.train_loader:  # Sử dụng train_loader của tác vụ vừa hoàn thành
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    # Sử dụng LoRA của tác vụ hiện tại (vừa học xong) để tính prototype cuối cùng
                    _, activations = self.net(inputs, return_activations=True, active_lora_task_id=self.current_task)
                    feat_norm = F.normalize(activations['feat'], p=2, dim=1)

                    for class_label_tensor in torch.unique(labels):
                        class_label_idx = class_label_tensor.long()
                        mask = (labels == class_label_tensor)
                        current_task_op_sum[class_label_idx] += feat_norm[mask].sum(dim=0)
                        current_task_sample_counts[class_label_idx] += mask.sum()

            newly_learned_in_this_task = []
            for class_label_idx in range(num_classes_dict[self.args.dataset]):
                if current_task_sample_counts[class_label_idx] > 0:
                    if class_label_idx not in self.learned_classes:
                        self.learned_classes.append(class_label_idx)
                        newly_learned_in_this_task.append(class_label_idx)

                    # Cập nhật self.op. Nếu muốn EMA, logic sẽ khác.
                    # Ở đây, ghi đè/thiết lập prototype bằng giá trị từ tác vụ hiện tại.
                    self.op[class_label_idx] = current_task_op_sum[class_label_idx] / current_task_sample_counts[
                        class_label_idx]
                    # Đồng thời cập nhật op_sum và sample_counts toàn cục nếu SARL gốc dựa vào chúng để tính lại sau này
                    self.op_sum[class_label_idx] = current_task_op_sum[class_label_idx].clone()  # Hoặc cộng dồn
                    self.sample_counts[class_label_idx] = current_task_sample_counts[
                        class_label_idx].clone()  # Hoặc cộng dồn

            print(
                f"  Finalized prototypes for classes: {newly_learned_in_this_task if newly_learned_in_this_task else 'None new this task (prototypes updated)'}")
            print(f"  All learned classes so far: {sorted(self.learned_classes)}")

            # 5. Chuẩn bị cho tác vụ tiếp theo
            self.current_task += 1  # Quan trọng: tăng current_task SAU KHI đã lưu LoRA của tác vụ cũ
            # self.eval_prototypes_for_lsm = True # Reset ở begin_task của tác vụ mới

            if self.args.save_interim:
                # Logic lưu trữ model và prototypes
                # torch.save(self.net.state_dict(), ...)
                # torch.save(self.op, ...)
                pass