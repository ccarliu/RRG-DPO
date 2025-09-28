import torch.nn.functional as F
import torch

def simpo_loss(
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        loss_type = "sigmoid",
        beta = 2, 
        gamma_beta_ratio = 0.5,
        label_smoothing = 0,
    ):
    # -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the SimPO loss for a batch of policy model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the SimPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        pi_logratios = pi_logratios.to(policy_chosen_logps.device)
        logits = pi_logratios - gamma_beta_ratio

        if loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(beta * logits) * (1 - label_smoothing)
                - F.logsigmoid(beta * logits) * label_smoothing
            )
        elif loss_type == "hinge":
            losses = torch.relu(1 - beta * logits)
        else:
            raise ValueError(
                f"Unknown loss type: {loss_type}. Should be one of ['sigmoid', 'hinge']"
            )

        chosen_rewards = beta * policy_chosen_logps.to(policy_chosen_logps.device).detach()
        rejected_rewards = beta * policy_rejected_logps.to(policy_chosen_logps.device).detach()

        return losses, chosen_rewards, rejected_rewards

def compute_logprobs(logits, labels, selection_mask=None):
    """
    Compute log probabilities.

    Args:
      logits: Tensor of shape (batch_size, num_tokens, vocab_size)
      labels: Tensor of shape (batch_size, num_tokens)
      selection_mask: Tensor for shape (batch_size, num_tokens)

    Returns:
      mean_log_prob: Mean log probability excluding padding tokens.
    """
    # print(logits.shape, labels.shape, selection_mask.shape)
    # Labels are the inputs shifted by one
    labels = labels[:, 1:].clone()
    # print(selection_mask)
    # Truncate logits to match the labels num_tokens
    logits = logits[:, :-1, :]

    log_probs = F.log_softmax(logits, dim=-1)

    # Gather the log probabilities for the actual labels
    selected_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)
    # print(logits.shape, labels.shape, selection_mask.shape)
    if selection_mask is not None:
        mask = selection_mask[:, 1:].clone()

        # Apply the mask to filter out padding tokens
        selected_log_probs = selected_log_probs * mask

        # Calculate the average log probability excluding padding tokens
        # This averages over the tokens, so the shape is (batch_size, num_tokens)
        avg_log_prob = selected_log_probs.sum(-1) / mask.sum(-1)

        return avg_log_prob

    else:
        return selected_log_probs.mean(-1)

def compute_dpo_loss(model_chosen_logprobs: torch.FloatTensor,
                    model_rejected_logprobs: torch.FloatTensor,
                    reference_chosen_logprobs: torch.FloatTensor,
                    reference_rejected_logprobs: torch.FloatTensor,
                    model_rej_p: torch.FloatTensor = None,
                    reference_rej_p: torch.FloatTensor = None,
                    beta: float = 0.1,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False):
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = model_chosen_logprobs - model_rejected_logprobs
    ref_logratios = reference_chosen_logprobs - reference_rejected_logprobs

    if reference_free:
        ref_logratios = 0

    if model_rej_p is None:
        penalty = 0.1 * torch.clamp(reference_chosen_logprobs - model_chosen_logprobs, min=0)
    else:
        penalty = 0.1 * torch.clamp(reference_chosen_logprobs - model_chosen_logprobs, min=0) + 0 * torch.clamp(reference_rej_p - model_rej_p, min=0)

    logits = pi_logratios - ref_logratios - penalty  # also known as h_{\pi_\theta}^{y_w,y_l}

    # logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (model_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards = beta * (model_rejected_logprobs - reference_rejected_logprobs).detach()

    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()


def compute_dpo_loss_(
      model_chosen_logprobs,
      model_rejected_logprobs,
      reference_chosen_logprobs,
      reference_rejected_logprobs,
      beta=0.1,
    ):
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logprobs: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logprobs: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logprobs: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logprobs: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss; typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss.

    Returns:
        A tuple of three tensors: (loss, chosen_rewards, rejected_rewards).
    """

    model_logratios = model_chosen_logprobs - model_rejected_logprobs
    reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
    logits = model_logratios - reference_logratios

    # DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
    losses = -F.logsigmoid(beta * logits)

    # Optional values to track progress during training
    chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach()

    # .mean() to average over the samples in the batch
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()