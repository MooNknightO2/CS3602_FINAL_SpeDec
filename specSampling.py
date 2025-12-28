import torch
from tqdm import tqdm
from utils import norm_logits, sample, max_fn

@torch.no_grad()
def specSampling(prefix: torch.Tensor, q_model: torch.nn.Module,
                 p_model: torch.nn.Module, maxLen: int, gamma: int,
                 temperature: float = 1, top_k: int = 0, top_p: float = 0) -> torch.Tensor:
    """
    Args:
        prefix: input sequence, (batch = 1, prefix_seqLen)
        q_model: approx model, the small one
        p_model: target model, the large one
        max_len: the max overall generated tokens number.
        gamma: the token number small model guesses.
        temperature (optional): Defaults to 1.
        top_k (optional): Defaults to 0.
        top_p (optional): Defaults to 0.
    Returns:
        torch.Tensor: generated tokens (batch, target_seqLen)
    """

    seqLen = prefix.shape[1]
    T = seqLen + maxLen
    assert prefix.shape[0] == 1
    with tqdm(total=T, desc="specSampling") as pbar:
        while prefix.shape[1] < T:
            x = prefix
            preLen = prefix.shape[1]
            for _ in range(gamma):
                q = q_model(x).logits
                next_tok = sample(norm_logits(q[:, -1, :], temperature, top_k, top_p))
                x = torch.cat((x, next_tok), dim=1)
            for i in range(q.shape[1]):
                q[:, i, :] = norm_logits(q[:, i, :], temperature, top_k, top_p)
            p = p_model(x).logits
            for i in range(p.shape[1]):
                p[:, i, :] = norm_logits(p[:, i, :], temperature, top_k, top_p)
            flag = True
            n = preLen - 1
            for i in range(gamma):
                r = torch.rand(1, device=p.device)
                j = x[:, preLen + i]
                if r < torch.min(torch.tensor([1], device=q.device), 
                                 p[:, preLen + i - 1, j] / q[:, preLen + i - 1, j]):
                    n += 1
                else:
                    t = sample(max_fn(p[:, n, :] - q[:, n, :]))
                    flag = False
                    break
            prefix = x[:, :(n + 1)]
            if flag:
                t = sample(p[:, -1, :])
            prefix = torch.cat((prefix, t), dim = 1)
            pbar.update(n - pbar.n)
    return prefix