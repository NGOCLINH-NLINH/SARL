import os


# best_params = {
#     200: {
#         'idt': 'v1',
#         'alpha': 0.5,
#         'beta': 1,
#         'op_weight': 0.5,
#         'sim_thresh': 0.8,
#         'sm_weight': 0.01,
#         'kw': '0.9 0.9 0.9 0.9',
#         'lr': 0.03,
#         'minibatch_size': 32,
#         'batch_size': 32,
#         'n_epochs': 50,
#         'lr_steps': '35 45',
#         'warmup_epochs': 3
#     },
#     500: {
#         'idt': 'v1',
#         'alpha': 0.2,
#         'beta': 1,
#         'op_weight': 0.5,
#         'sim_thresh': 0.8,
#         'sm_weight': 0.01,
#         'kw': '0.9 0.9 0.9 0.9',
#         'lr': 0.03,
#         'minibatch_size': 32,
#         'batch_size': 32,
#         'n_epochs': 50,
#         'lr_steps': '35 45',
#         'warmup_epochs': 3
#     },
# }
#
#
# # lst_seed = [1, 3, 5]
# lst_seed = [1]
# # lst_buffer_size = [200, 500]
# lst_buffer_size = [200]
# count = 0
# output_dir = "experiments/sarl"
# save_model = 0  # set to 1 to save the final model
# save_interim = 0  # set to 1 to save intermediate model state and running params
# device = 'mps'
#
# for seed in lst_seed:
#     for buffer_size in lst_buffer_size:
#         params = best_params[buffer_size]
#         exp_id = f"sarl-cifar100-{buffer_size}-param-{params['idt']}-s-{seed}"
#         job_args = f"python main.py  \
#             --experiment_id {exp_id} \
#             --model sarl \
#             --dataset seq-cifar100 \
#             --kw {params['kw']} \
#             --alpha {params['alpha']} \
#             --beta {params['beta']} \
#             --op_weight {params['op_weight']} \
#             --sim_thresh {params['sim_thresh']} \
#             --sm_weight {params['sm_weight']} \
#             --buffer_size {buffer_size} \
#             --batch_size {params['batch_size']} \
#             --minibatch_size {params['minibatch_size']} \
#             --lr {params['lr']} \
#             --lr_steps {params['lr_steps']} \
#             --n_epochs {params['n_epochs']} \
#             --output_dir {output_dir} \
#             --csv_log \
#             --seed {seed} \
#             --device {device} \
#             --save_model {save_model} \
#             --save_interim {save_interim} \
#             "
#         count += 1
#         os.system(job_args)
#
# print('%s jobs counted' % count)

import os

best_params = {
    200: {
        'idt': 'v1_drs',  # Thay đổi idt để phân biệt
        'alpha': 0.5, 'beta': 1.0, 'op_weight': 0.5, 'sim_thresh': 0.8, 'sm_weight': 0.01,
        'kw': '0.9 0.9 0.9 0.9', 'lr': 0.03, 'minibatch_size': 32, 'batch_size': 32,
        'n_epochs': 50, 'lr_steps': '35 45', 'warmup_epochs': 3,
        # --- THÊM SIÊU THAM SỐ CHO LoRA/DRS/ATL ---
        'lora_rank': 8, 'drs_variance_threshold': 0.95, 'lambda_atl': 0.1, 'atl_margin': 0.2,
        'max_samples_for_drs': 512, 'num_tasks': 10  # Giả sử CIFAR-100 chia thành 10 tác vụ
    },
    500: {
        'idt': 'v1_drs',
        'alpha': 0.2, 'beta': 1.0, 'op_weight': 0.5, 'sim_thresh': 0.8, 'sm_weight': 0.01,
        'kw': '0.9 0.9 0.9 0.9', 'lr': 0.03, 'minibatch_size': 32, 'batch_size': 32,
        'n_epochs': 50, 'lr_steps': '35 45', 'warmup_epochs': 3,
        # --- THÊM SIÊU THAM SỐ CHO LoRA/DRS/ATL ---
        'lora_rank': 8, 'drs_variance_threshold': 0.95, 'lambda_atl': 0.1, 'atl_margin': 0.2,
        'max_samples_for_drs': 512, 'num_tasks': 10
    },
}

lst_seed = [1]
lst_buffer_size = [200]  # Thử với một buffer size trước
count = 0
output_dir = "experiments/sarl_drs_test"  # Thay đổi output_dir để không ghi đè
save_model = 0
save_interim = 0
device = 'cuda'  # Hoặc 'cuda' hoặc 'cpu'

for seed in lst_seed:
    for buffer_size in lst_buffer_size:
        params = best_params[buffer_size]
        # --- THAY ĐỔI TÊN MODEL TRONG EXP_ID ---
        exp_id = f"sarl_drs-cifar100-{buffer_size}-param-{params['idt']}-s-{seed}"

        job_args_list = [
            "python main.py",  # Tên file main của bạn
            f"--experiment_id {exp_id}",
            "--model sarl_drs",  # <<< THAY ĐỔI QUAN TRỌNG
            "--dataset seq-cifar100",
            f"--kw \"{params['kw']}\"",  # Đặt trong dấu nháy kép nếu có khoảng trắng
            f"--alpha {params['alpha']}",
            f"--beta {params['beta']}",
            f"--op_weight {params['op_weight']}",
            f"--sim_thresh {params['sim_thresh']}",
            f"--sm_weight {params['sm_weight']}",
            f"--buffer_size {buffer_size}",
            f"--batch_size {params['batch_size']}",
            f"--minibatch_size {params['minibatch_size']}",
            f"--lr {params['lr']}",
            f"--lr_steps \"{params['lr_steps']}\"",  # Đặt trong dấu nháy kép
            f"--n_epochs {params['n_epochs']}",
            f"--warmup_epochs {params['warmup_epochs']}",  # Thêm warmup_epochs
            f"--output_dir {output_dir}",
            "--csv_log",
            f"--seed {seed}",
            f"--device {device}",
            f"--save_model {save_model}",
            f"--save_interim {save_interim}",
            # --- THÊM TRUYỀN CÁC SIÊU THAM SỐ MỚI ---
            f"--lora_rank {params['lora_rank']}",
            f"--drs_variance_threshold {params['drs_variance_threshold']}",
            f"--lambda_atl {params['lambda_atl']}",
            f"--atl_margin {params['atl_margin']}",
            f"--max_samples_for_drs {params['max_samples_for_drs']}",
            f"--num_tasks {params['num_tasks']}"  # Truyền num_tasks
        ]
        job_args = " ".join(job_args_list)

        print("=" * 50)
        print(f"RUNNING JOB: {exp_id}")
        print(job_args)
        print("=" * 50)

        count += 1
        os.system(job_args)

print('%s jobs counted' % count)
