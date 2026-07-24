import numpy as np
import torch

def load_dpo_data(path, shape):
    data = np.memmap(path, dtype=np.int32, mode='r')
    return data.reshape(shape)

def get_dpo_batch(data, batch_size, device):
    """
    data: (n_samples, 2, seq_len)
    取 batch 个 SFT 样本，把 assistant 截断做 rejected
    """
    n = data.shape[0]
    indices = torch.randint(0, n, (batch_size,))

    chosen_ids_list, chosen_labels_list = [], []
    rejected_ids_list, rejected_labels_list = [], []

    for idx in indices:
        input_ids = data[idx, 0].astype(np.int64)
        labels = data[idx, 1].astype(np.int64)

        # 找到 answer 区间（labels != -100 的位置）
        answer_mask = (labels != -100)
        answer_positions = np.where(answer_mask)[0]

        if len(answer_positions) < 4:
            continue  # 回答太短跳过

        # chosen = 原样
        chosen_ids = input_ids.copy()
        chosen_labels = labels.copy()

        # rejected = 回答截断一半
        cutoff = answer_positions[len(answer_positions) // 2]
        rejected_ids = input_ids.copy()
        rejected_labels = labels.copy()
        rejected_labels[cutoff:] = -100

        chosen_ids_list.append(chosen_ids)
        chosen_labels_list.append(chosen_labels)
        rejected_ids_list.append(rejected_ids)
        rejected_labels_list.append(rejected_labels)

    return (
        torch.tensor(np.stack(chosen_ids_list), dtype=torch.long).to(device),
        torch.tensor(np.stack(chosen_labels_list), dtype=torch.long).to(device),
        torch.tensor(np.stack(rejected_ids_list), dtype=torch.long).to(device),
        torch.tensor(np.stack(rejected_labels_list), dtype=torch.long).to(device),
    )