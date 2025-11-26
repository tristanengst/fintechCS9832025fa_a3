import argparse
from datetime import datetime, timedelta
import copy
import einops
from functools import wraps
import numpy as np
import os
import os.path as osp
import peft
from peft import PeftModel
import random
import time
import torch
import torch.nn as nn
from tqdm import tqdm

import UtilsBase

####### Utilities to support faster-than-vanilla-GPU training ########################

def twrite(*args, quiet=False, **kwargs):
    """Better version of print() and tqdm.write().

    Args:
    *args       -- arguments to print (analogous to print())
    quiet       -- if True, suppresses all printing
    **kwargs    -- key-value pairs to print
    """
    if quiet:
        return
    args_str = ",".join([str(a) for a in args])
    kwargs_str = ",".join([f"{k}={v}" for k,v in kwargs.items()])
    s = args_str + ("," if kwargs_str and args_str else "") + kwargs_str
    s = f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}] " + s
    _ = tqdm.write(s)

def disable_grads(model):
    """Returns [model] with all its gradients disabled."""
    for param in model.parameters():
        param.requires_grad = False
    return model

def wrap_model(*, model, args, lora_inference_mode=False, **kwargs):
    """Returns a version of [model] 'wrapped' in some way to possibly speed it up.
    
    Args:
    model                       -- model to wrap
    args                        -- Namespace with arguments
    find_unused_parameters      -- whether to set find_unused_parameters=True in DDP
    kwargs                      -- extra keyword arguments to use in [args]
    """
    args = UtilsBase.updated_namespace(args, **kwargs)
    if args.speedup == "compile":
        model = torch.compile(model, **kwargs)
    elif args.speedup == "gpu":
        model = model.to(get_device(args))
    elif args.speedup == "lora": # Maybe bad idea!!! Things break. I have not tested saving/resuming with this either.
        lora_cfg = peft.LoraConfig(r=32, lora_alpha=32,
            target_modules=["q_proj","v_proj"],
            modules_to_save=["head"] if hasattr(model, "head") else None,
            inference_mode=lora_inference_mode,
            bias="none",
            task_type="CAUSAL_LM",)
        model = peft.get_peft_model(model, lora_cfg)
    else:
        raise NotImplementedError()
    return model

def unwrap_model(model):
    """Returns [model] unwrapped, thereby exposing what's inside the wrapper."""
    if isinstance(model, nn.DataParallel | nn.parallel.DistributedDataParallel):
        return unwrap_model(model.module)
    elif isinstance(model, PeftModel):
        return unwrap_model(model.base_model)
    else:
        return model

def get_device(args=None):
    """Returns the device for the current process."""
    if not args is None and hasattr(args, "gpu") and torch.cuda.is_available():
        return torch.device(f"cuda:{args.gpu}")
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################################
######################################################################################
######################################################################################



####### Functions for keeping track of randomness ####################################

def get_seed_dict():
    """Returns a dictionary giving the seeds when the function is called."""
    return dict(random_seed=random.getstate(),
        torch_seed=torch.get_rng_state(),
        torch_cuda_seed=torch.cuda.get_rng_state(),
        numpy_seed=np.random.get_state())

def set_seed(seed, verbose=False):
    """Seeds the program to use seed [seed]."""
    if isinstance(seed, int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        _ = twrite(f"Set seed={seed}") if verbose else None
    elif isinstance(seed, dict):
        random.setstate(seed["random_seed"])
        np.random.set_state(seed["numpy_seed"])
        torch.set_rng_state(seed["torch_seed"])
        torch.cuda.set_rng_state(seed["torch_cuda_seed"])
        _ = twrite(f"Set seed to old seed from dict") if verbose else None
    else:
        raise ValueError(f"Seed should be int or contain resuming keys")
    return seed

def stateless(fn):
    """Decorator to do a function in a stateless way. Some things with eg.
    multiprocessing are surprisingly and annoyingly non-stateless! Note that seeds
    used deliberately inside the function can still change state! This function is
    intended to allow not worrying about this.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with SeedContextManager("cur_seed"):
            return fn(*args, **kwargs)
    return wrapper

class SeedContextManager:
    """Context manager for forcing all seeds to a value at the start of the context,
    and setting them to what they were prior to the context starting immediately upon
    its end.

    Args:
    seed    -- an integer seed or seed state dict, or None. In the latter case, using
                the SeedContextManager is a no-op
    name    -- optional name for the SeedContextManager for debugging purposes
    """
    def __init__(self, seed=None, name=None):
        self.seed = get_seed_dict() if seed == "cur_seed" else seed
        self.old_seed_cur_state = None
        self.seed_cur_state = None
        self.name = name

    def __enter__(self):
        if not self.seed is None:
            self.old_seed_cur_state = get_seed_dict()
            seed = self.seed if self.seed_cur_state is None else self.seed_cur_state
            _ = set_seed(seed, verbose=False)

    def __exit__(self, type, value, traceback):
        self.seed_cur_state = get_seed_dict()
        if not self.old_seed_cur_state is None:
            _ = set_seed(self.old_seed_cur_state, verbose=False)

    def __str__(self):
        return f"{self.__class__.__name__} [name={'' if self.name is None else self.name}, cur_randbits={self.getrandbits_nochange()}]"

    def state_str_nochange(self, digits_per_lib=4):
        with self:
            return state_to_str(digits_per_lib=digits_per_lib)

    def getrandbits(self, k=32):
        """Seeded version fo random.getrandbits()"""
        with self:
            return random.getrandbits(k)

    def getrandbits_nochange(self, k=32):
        with self:
            with SeedContextManager("cur_seed"):
                return random.getrandbits(k)

    def state_dict(self): return dict(seed=self.seed,
        seed_cur_state=self.seed_cur_state,
        old_seed_cur_state=self.old_seed_cur_state)

    def load_state_dict(self, state_dict):
        state_dict = torch.load(state_dict, weights_only=False) if isinstance(state_dict, str) else state_dict
        if self.seed == state_dict["seed"]:
            self.seed_cur_state = state_dict["seed_cur_state"]
            self.old_seed_cur_state = state_dict["old_seed_cur_state"]
        else:
            raise ValueError(f"Can not load SeedContextManager state as the start seed {self.seed} does not match the seed of the SeedContextManager being loaded: {state_dict['seed']}")

@stateless
def state_to_str(digits_per_lib=4):
    multiplier = 10 ** digits_per_lib
    torch_int = int(torch.rand(1).item() * multiplier)
    torch_cuda_int = int(torch.rand(1, device="cuda").item() * multiplier)
    np_int = int(np.random.rand() * multiplier)
    rand_int = int(random.random() * multiplier)
    return f"{rand_int}.{np_int}.{torch_int}.{torch_cuda_int}"

######################################################################################
######################################################################################
######################################################################################


########## Saving and loading models #################################################
def experiment_name(args, make_dir=False):
    """Returns the name of the experiment being run with [args]. If [make_dir] is set,
    the folder is created on disk.
    """
    def value_to_str(v):
        """Returns a string representation of [v] suitable for inclusion in experiment
        names, which may be filesystem paths.
        """
        if isinstance(v, list | tuple):
            return "_" + "_".join([value_to_str(vv) for vv in v])
        elif isinstance(v, float):
            return f"{v:.3f}" if abs(v) >= 1e-3 else f"{v:.2e}"
        elif isinstance(v, str):
            return f"_{v}"
        else:
            return str(v)

    def arg_to_pretty_name(arg):
        """Returns a shortened pretty name for command-line argument [arg_name]."""
        arg_name2pretty_name = dict(epochs_warmup="epsW", grpo_completions="grpoC",
            grpo_completions_expf="grpoCExpF", grpo_max_new_tokens="grpoMNT",
            grpo_kl_weight="grpoKL", grpo_clip="grpoClip", tr_update_iter="trU",
            tr_lr="trLR", tr_epochs_per_epoch="trEPS",
        )
        return arg_name2pretty_name[arg] if arg in arg_name2pretty_name else arg

    exclude_from_naming = ["naming", "suffix", "uid", "default"]
    default_naming = ["bs", "epochs", "lr", "grpo_completions", "grpo_max_new_tokens",]
    naming = args.naming + (default_naming if "default" in args.naming else [])
    naming = [n for n in naming if not n in exclude_from_naming]
    naming = sorted(set(naming))
    naming_kv = {k: value_to_str(vars(args)[k]) for k in naming}
    naming_str = "-".join([f"{k}{v}" for k,v in naming_kv.items()])
    suffix_str = f"-{args.suffix}" if not args.suffix is None else ""
    # Always use grpo prefix, then hyperparameters, then uid, then suffix
    exp_name = f"grpo-{naming_str}-{args.uid}{suffix_str}" 

    exp_dir_path = osp.join(args.save_dir, exp_name)
    _ = os.makedirs(exp_dir_path, exist_ok=True) if make_dir else None
    return exp_name

def save_all(*, epoch, args, save_latest=False, save_seeds=True, prefix="grpo_checkpoint", **kwargs):
    """Saves a checkpoint containing [kwargs] as a dictionary to disk."""
    to_save = dict(epoch=epoch, args=args, **kwargs)
    to_save = to_save | (get_seed_dict() if save_seeds else {})
    to_save = {k: vars(v) if isinstance(v, argparse.Namespace) else v for k,v in to_save.items()}   # Namespace to dict
    to_save = {k: unwrap_model(v) if isinstance(v, nn.Module) else v for k,v in to_save.items()}    # Unwrap models
    to_save = {k: v.state_dict() if hasattr(v, "state_dict") else v for k,v in to_save.items()}     # State dicts

    exp_dir = osp.join(args.save_dir, experiment_name(args, make_dir=True))
    fname = f"{prefix}_epoch{epoch}_latest.pt" if save_latest else f"{prefix}_epoch{epoch}.pt"
    save_start_time = time.time()
    save_path = osp.join(exp_dir, fname)
    _ = torch.save(to_save, save_path)
    
    if save_latest:
        latest_files = [f for f in os.listdir(exp_dir) if f.startswith(prefix) and f.endswith("latest.pt") and not f == fname]
        for f in latest_files:
            os.remove(osp.join(exp_dir, f))
    
    elapsed_time = (time.time() - save_start_time) / 60
    _ = twrite(f"elapsed time={elapsed_time:1}m Saved latest checkpoint to {save_path}")

def resume_all(*, resume_path, **kwargs):
    """Returns a dictionary loaded from checkpoint at [resume_path]. Any keys in the
    dictionary matching those in [kwargs] will have their state-dicts loaded from the
    saved values or otherwise set equal to those in [kwargs].
    """
    checkpoint = torch.load(resume_path, map_location=get_device(), weights_only=False)
    result = dict()
    for k,v in kwargs.items():
        if k in checkpoint and hasattr(v, "load_state_dict"):
            v.load_state_dict(checkpoint[k])
            result[k] = v
        elif k in checkpoint:
            result[k] = checkpoint[k]
        else:
            result[k] = v
    return result

def args_to_latest_checkpoint(args, prefix="grpo_checkpoint"):
    """Returns the path to the latest checkpoint for the experiment specified by
    [args], or None if no such checkpoint exists.
    """
    exp_dir = osp.join(args.save_dir, experiment_name(args, make_dir=False))
    if not osp.exists(exp_dir):
        return None
    completion_files = [f for f in os.listdir(exp_dir) if f.startswith(prefix) and f.endswith(".pt")]
    if len(completion_files) == 0:
        return None
    file2epoch = {f: f.split("_epoch")[1].split("_")[0] for f in completion_files}
    file2epoch = {f: int(e.replace("latest.pt","").replace(".pt","")) for f,e in file2epoch.items()}
    latest_file = max(file2epoch.items(), key=lambda x: x[1])[0]
    return osp.join(exp_dir, latest_file)

######################################################################################
######################################################################################
######################################################################################
def mask_unmasked_to_the_left(attention_mask):
    """Returns a BSxT mask matching [attention_mask] but where in each sequence, the
    position immediately to the left of the leftmost unmasked token is also unmasked
    unless the leftmost unmasked token is in position 0, or the entire sequence is
    masked.
    """
    bs, t = attention_mask.shape
    flipped_mask = (~attention_mask.bool()).float()
    first_unmasked_pos = torch.cumprod(flipped_mask, dim=1)
    first_unmasked_pos = einops.reduce(first_unmasked_pos, "bs t -> bs 1", "sum").long() - 1
    first_unmasked_pos[first_unmasked_pos == t-1] = t # If all masked, nothing is unmasked
    t_idxs = torch.arange(t, device=attention_mask.device)
    t_idxs = einops.rearrange(t_idxs, "t -> 1 t")
    result = torch.where(first_unmasked_pos == t_idxs, 1, attention_mask)
    return result.to(attention_mask.dtype)


class LinearRampThenCosineAnnealingScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Learning rate scheduler that linearly ramps up the learning rate from
    [start_lr] to [lr] over [warmup_epochs], and then cosine anneals from [lr] to
    [end_lr] over the remaining epochs up to [total_epochs].

    The scheduler's step should be tracked externally and passed in. This promotes
    easier alignment/logging of the learning rate schedule with training steps.
    Essentially it should be treated as being stateless.

    Args:
    optimizer       -- optimizer to schedule (its internal learning rate is ignored!)
    warmup_epochs   -- number of epochs to linearly ramp up the learning rate
    total_epochs    -- total number of epochs for the schedule
    last_epoch      -- last epoch number (for resuming)
    start_lr        -- starting learning rate at epoch 0
    lr              -- learning rate after warmup
    end_lr          -- ending learning rate at total_epochs
    steps_per_epoch -- number of steps per epoch
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1, start_lr=0.0, lr=1.0, end_lr=0.0, steps_per_epoch=1):
        self.last_epoch = last_epoch
        self.warmup_epochs = warmup_epochs
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_epochs = total_epochs
        self.total_steps = total_epochs * steps_per_epoch
        self.cosine_steps = self.total_steps - self.warmup_steps
        self.start_lr = start_lr
        self.lr = lr
        self.end_lr = end_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self, step=None):
        step = self.last_epoch if step is None else step
        if step <= self.warmup_steps:
            warmup_progress = step / self.warmup_steps
            return [self.start_lr + warmup_progress * (self.lr - self.start_lr) for _ in self.base_lrs]
        else:
            progress = (step - self.warmup_steps) / self.cosine_steps
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return [self.end_lr + (self.lr - self.end_lr) * cosine_decay for _ in self.base_lrs]

    def step(self, step=None):
        """Step the scheduler to have the learning rate it would have at [step]. It is
        generally better/easier to track the step count explicitly and pass it in here
        rather than to rely on a stateful (very evil) internal counter.
        """
        if step is None:
            self.last_epoch += 1
        else:
            self.last_epoch = step
        step = self.last_epoch
        super(LinearRampThenCosineAnnealingScheduler, self).step(epoch=step)