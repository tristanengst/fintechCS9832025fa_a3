"""Utility functions I think should be in Python's standard library."""
import argparse
import copy
from collections import defaultdict
from datetime import datetime
import functools
from functools import partial
import json
import math
import os
import os.path as osp
import uuid
import time

try:
    from tqdm import tqdm
except ImportError:
    class tqdm_lite:
        """Stand-in for tqdm if it is not installed."""
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
        def __iter__(self): return iter(self.iterable)
        def write(s): print(s)
    tqdm = tqdm_lite


class NoOp:
    def __getattr__(self, name): return self
    def __call__(self, *args, **kwargs): return self

try:
    import torch
    torch_dtypes = [torch.float32, torch.int32, torch.bool,
        torch.float64, torch.int64, torch.int16, torch.uint8, torch.bfloat16]
except ImportError:
    torch = NoOp()
    torch_dtypes = []

######################################################################################
######################################################################################
######################################################################################

####### I/O Functions ################################################################

from_kwargs_types = [argparse.Namespace]
def to_json_encoding(x):
    """Returns a JSON-serializable dictionary representation of [x] for common types
    we might want to serialize and deserialize. Useful exactly to the extent we don't
    trust pickling.
    """
    if isinstance(x, torch.Tensor) and x.dtype in torch_dtypes:
        assert x.grad is None, "to_json: tensors with gradients not supported"
        return dict(type=str(x.dtype), data=x.cpu().tolist())
    elif isinstance(x, list | tuple | set):
        return type(x)([to_json_encoding(v) for v in x.items()])
    elif isinstance(x, dict):
        return {k: to_json_encoding(v) for (k, v) in x.items()}
    elif isinstance(x, int | float | bool | str):
        return x
    elif isinstance(x, tuple(from_kwargs_types)):
        return dict(type=type(x).__name__, data={k: to_json_encoding(v) for (k, v) in vars(x).items()})
    else:
        raise ValueError(f"to_json: Cannot serialize object of type {type(x)}")

def from_json_encoding(x, device="cpu", **kwargs):
    """Returns a deserialized object from its JSON-serialized dictionary
    representation, with [kwargs] passed to its constructor or those of any objects
    lying within it.

    Args:
    x       -- JSON-deserialized object (eg. from json.load())
    device  -- device to place any torch.Tensors
    """
    type2constructor = {t.__name__: t for t in [int, float, bool, str, list, tuple, dict, set]}
    type2constructor |= {str(td): partial(torch.tensor, device=device, dtype=td) for td in torch_dtypes}
    type2constructor |= {str(t): lambda x: t(**{k: from_json_encoding(v) for k,v in x.items()}) for t in from_kwargs_types}

    if isinstance(x, dict) and set(x.keys()) == {"type", "data"} and x["type"] in type2constructor:
        return type2constructor[x["type"]](x["data"], **kwargs)
    elif isinstance(x, dict):
        return {k: from_json_encoding(v, **kwargs) for (k, v) in x.items()}
    elif isinstance(x, list | tuple | set):
        return type(x)([from_json_encoding(v, **kwargs) for v in x])
    elif isinstance(x, int | float | bool | str):
        return x
    else:
        raise ValueError(f"from_json: Cannot deserialize object of type {type(x)}")


def twrite(*args, time=True, verbose=1, quiet=False, offset=False, **kwargs):
    """Lite version of twrite(). Doesn't support multiple processes."""
    if quiet or verbose < 1:
        return

    def pretty_time(offset=False):
        offset = " " * 6 if offset else ""
        return f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}]{offset}"
    
    def pretty_time_space(): return pretty_time(offset=True)

    def separated_str(*strs): return " ".join([s for s in strs if not s == ""])

    meta_str = f"{pretty_time(offset=offset)}" if time else (" " * 6 if offset else "")
    kwargs_str = " ".join([f"{k}={v}" for k,v in kwargs.items()])
    args_str = " ".join([str(a) for a in args])
    s = separated_str(meta_str, args_str, kwargs_str)
    tqdm.write(s)

def load_file_lite(fname, json_kwargs=dict(), **kwargs):
    """Loads a file [fname] with [kwargs]. The kind of load function is inferred from
    the file extension. This version does not support .pt files.
    """
    with open(fname, "r") as f:
        if fname.endswith(".pt"):
            raise NotImplementedError("Loading .pt files is not supported")
        elif fname.endswith(".json"):
            return json.load(f, **json_kwargs)
        elif fname.endswith(".txt") or fname.endswith(".sh") or fname.endswith(".py"):
            return f.read()
        else:
            raise NotImplementedError(f"Unknown file extension for {fname}")

def atomic_save_lite(*, data, fname, **kwargs):
    """Atomically saves [data] to [fname] with [kwargs]. The kind of save function is
    inferred from the file extension. This version does not support .pt files.
    """
    _ = os.makedirs(osp.dirname(fname), exist_ok=True) if osp.dirname(fname) else None
    fname_base, ext = osp.splitext(fname)
    tmp_file = f"__tempfile__{str(uuid.uuid4()).replace('-', '')}_{osp.basename(fname_base)}.tmp"
    tmp_file = osp.join(osp.dirname(fname), tmp_file)

    if fname.endswith(".pt"):
        raise NotImplementedError("Saving .pt files is not supported")
    elif fname.endswith(".json"):
        kwargs["indent"] = 4 if not "indent" in kwargs else kwargs["indent"]
        with open(tmp_file, "w+") as f:
            json.dump(data, f, **kwargs)
    elif fname.endswith(".txt") or fname.endswith(".sh") or fname.endswith(".py"):
        with open(tmp_file, "w+") as f:
            f.write(data)
    else:
        raise NotImplementedError(f"Unknown file extension for {fname}")
    os.rename(tmp_file, fname)

def atomic_append_lite(*, data, fname, **kwargs):
    fname = osp.expanduser(fname)
    if fname.endswith(".pt"):
        raise NotImplementedError("Appending to .pt files is not supported")
    elif fname.endswith(".json"):
        return atomic_save_lite(data=load_file_lite(f) | d, fname=f, indent=4, sort_keys=True)
    elif fname.endswith(".txt") or fname.endswith(".sh") or fname.endswith(".py"):
        with open(fname, "a+") as f:
            f.write(data)
    else:
        raise NotImplementedError(f"Unknown file extension for {fname}")


def dict_to_json(d, f): return atomic_save_lite(data=d, fname=f, indent=4, sort_keys=True)
def json_to_dict(f): return load_file_lite(f)
def dict_append_json(d, f): atomic_append_lite(data=d, fname=f)

def path_from_home(f):
    """Returns a path to [f] that will work from any home directory."""
    abspath = osp.abspath(osp.expanduser(f))
    home = osp.abspath(osp.expanduser("~"))
    return f"~/{abspath[len(home)+1:]}"
    
######################################################################################
######################################################################################
######################################################################################


###### String Processing Functions ###################################################
def strip_right(s, remove):
    """Returns [s] with the substring [remove] removed from the right side of it if
    [s] ends with [remove], and [s] otherwise. Contrast with rstrip, which removes any
    character in [remove], which is not what I'd expect.
    """
    return s[:-len(remove)] if s.endswith(remove) else s

def strip_left(s, remove):
    """Returns [s] with the substring [remove] removed from the left side of it if
    [s] starts with [remove], and [s] otherwise. Contrast with lstrip, which removes any
    character in [remove], which is not what I'd expect.
    """
    return s[len(remove):] if s.startswith(remove) else s

def remove_nonnumeric(s):
    """Returns [s] with all non-numeric characters removed."""
    return "".join([c for c in s if c.isnumeric()])

######################################################################################
######################################################################################
######################################################################################

###### Data structure functions ######################################################

def updated_namespace(extant, *updated, **kwargs):
    """Returns a new argparse Namespace that updates [extant] with [updated] and [kwargs]."""
    assert len(updated) <= 1, "Only one updated argument is allowed"
    extant = vars(extant) if isinstance(extant, argparse.Namespace) else extant
    updated = dict() if len(updated) == 0 else updated[0]
    updated = vars(updated) if isinstance(updated, argparse.Namespace) else updated
    return argparse.Namespace(**extant | updated | kwargs)

def dict_to_namespace(d):
    """Returns possibly-nested dictionary [d] as an argparse Namespace."""
    d = vars(d) if isinstance(d, argparse.Namespace) else d
    if isinstance(d, dict):
        return argparse.Namespace(**{k: dict_to_namespace(v) for k,v in d.items()})
    elif isinstance(d, list | tuple): # Obviously this shouldn't be the outer call!
        return d.__class__([dict_to_namespace(v) for v in d])
    else:
        return d

def namespace_to_dict(n):
    """Returns the namespace [n] as a dictionary."""
    n = vars(n) if isinstance(n, argparse.Namespace) else n
    if isinstance(n, dict):
        return {k: namespace_to_dict(v) for k,v in n.items()}
    elif isinstance(n, list | tuple | set):
        return n.__class__([namespace_to_dict(v) for v in n])
    elif isinstance(n, str | int | float | bool | None):
        return n
    else:
        raise ValueError(f"Could not convert {n} to dictionary: {type(n)}")

def flatten(xs):
    """Returns collection [xs] after recursively flattening into a list."""
    type_map = {type({}.items()): list, type({}.values()): list, type({}.keys()): set}
    xs = type_map[type(xs)](xs) if type(xs) in type_map else xs

    if isinstance(xs, list | set | tuple):
        result = []
        for x in xs:
            result += flatten(x) if isinstance(x, list | set | tuple) else [x]
        return xs.__class__(result)
    else:
        return xs

######################################################################################
######################################################################################
######################################################################################


###### Time Functions ################################################################
def seconds_since_time(start_time):
    if isinstance(start_time, datetime):
        return (datetime.now() - start_time).total_seconds()
    elif isinstance(start_time, str):
        return (datetime.now() - time_stamp_to_datetime(start_time)).total_seconds()
    else:
        return time.time() - start_time  
def hours_since_time(start_time): return seconds_since_time(start_time) / 3600
def minutes_since_time(start_time): return seconds_since_time(start_time) / 60
def seconds_to_minutes(seconds): return seconds / 60
def seconds_to_hours(seconds): return seconds / 3600

def time_stamp_to_datetime(time_stamp):
    """Converts a time stamp to a datetime object."""
    if isinstance(time_stamp, datetime):
        return time_stamp
    
    time_stamp = time_stamp.strip()
    # Common custom time stamp format to make life easier
    if time_stamp.find("-") in [1,2]:
        dt = datetime.strptime(time_stamp, "%m-%d-%H:%M")
        dt = dt.replace(year=datetime.now().year)
        return dt
    elif "T" in time_stamp:
        return datetime.strptime(time_stamp, "%Y-%m-%dT%H:%M:%S")
    elif "-" in time_stamp:
        return datetime.strptime(time_stamp, "%Y-%m-%d-%H:%M:%S")
    else:
        raise ValueError(f"Could not parse time stamp: {time_stamp}")

def time_to_seconds(time_str):
    """Returns [time_str] as a number of seconds. Tries to fit as many possible ways
    [time_str] could be interpreted as a duration; it need not actually be a string.

    This sort of time string would indicate a duration.
    """
    if isinstance(time_str, int | float):
        return time_str

    time_str = time_str.strip()
    if time_str.lower().endswith("s"):
        return float(time_str[:-1])
    elif time_str.lower().endswith("m"):
        return float(time_str[:-1]) * 60
    elif time_str.lower().endswith("h"):
        return float(time_str[:-1]) * 3600
    elif time_str.lower().endswith("d"):
        return float(time_str[:-1]) * 24 * 3600
    elif "-" in time_str:
        days, time_str = time_str.split("-")
        
        # If there is only a single colon in [time_str] now, then assume that the
        # seconds are not included.
        time_str = f"{time_str}:00" if time_str.count(":") == 1 else time_str
        
        return int(days) * 24 * 3600 + time_to_seconds(time_str)
    # Usually output by SLURM. Assumes that seconds are present!
    elif ":" in time_str:
        times = time_str.split(":")
        return sum([int(t) * (60 ** idx) for idx,t in enumerate(reversed(times))])
    else:
        time_suffix2seconds = dict(H=3600, M=60, S=1, D=24*3600)
        s, cur_num = 0, ""
        for c in time_str:
            if c.isnumeric():
                cur_num += c
            elif c in time_suffix2seconds and cur_num:
                s += int(cur_num) * time_suffix2seconds[c.upper()]
                cur_num = ""
            else:
                raise ValueError(f"Invalid character in time string: {c} in {time_str}")
        return s
        
def time_to_hours(t): return time_to_seconds(t) / 3600
def time_to_minutes(t): return time_to_seconds(t) / 60
def time_to_str(t):
    """Returns time string [time_str] in our default way, ie. without days."""
    s = int(t) if isinstance(t, float | int) else time_to_seconds(t)
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f"{int(h)}:{int(m):02}:{int(s):02}"

def time_to_pretty_str(t):
    """Returns XXHYYM for XX hours and YY minutes. Days are collapsed to hours."""
    s = int(t) if isinstance(t, float | int) else time_to_seconds(t)
    h = s // 3600
    m = (s % 3600) // 60
    h = str(h).zfill(max(2, len(str(h))+1))
    return f"{h}H{m:02d}M"

######################################################################################
######################################################################################
######################################################################################


###### Persisted State ###############################################################
# For when environment variables have semantics not suitable for the task at hand. In
# this case, we will generally assume that either there is one process running per
# machine or that a machine isolates processes that would use this via $SLURM_TMPDIR.
def get_persisted_state_file():
    """Returns the path to the persisted state file."""
    return f"{os.environ['SLURM_TMPDIR'] if 'SLURM_JOB_ID' in os.environ else '.'}/persisted_state.json"

def persisted_state_get_all():
    """Returns all persisted state as a dictionary."""
    f = get_persisted_state_file()
    return json_to_dict(f) if osp.exists(f) else dict()

def persisted_state_get(k, default=None):
    """Returns the value of [k] in persisted state or [default] if it isn't found."""
    persisted_state = persisted_state_get_all()
    return persisted_state.get(k, (default() if callable(default) else default))

def persisted_state_contains(k):
    """Returns if [k] is in the persisted state."""
    return k in persisted_state_get_all()

def persisted_state_update(**kwargs):
    """Sets key [k] to value [v] in persisted state."""
    _ = dict_append_json(kwargs, get_persisted_state_file())

def persistent_state_del(k):
    """Deletes key [k] from the persisted state."""
    persisted_state = persisted_state_get_all()
    persisted_state = {k1: v for k1,v in persisted_state.items() if not k1 == k}
    return dict_to_json(persisted_state, get_persisted_state_file())

def persisted_state_clear():
    """Removes the persisted state file."""
    _ = dict_to_json(dict(), get_persisted_state_file())

######################################################################################
######################################################################################
######################################################################################

###### Misc Functions ################################################################

def int_divides(numerator, denominator):
    """Returns numerator // denominator if [denominator] divides [numerator] evenly,
    or otherwise raises a ValueError.
    """
    if not numerator % denominator == 0:
        raise ValueError(f"{denominator} does not divide {numerator} evenly")
    return numerator // denominator


