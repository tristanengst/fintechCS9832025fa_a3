import argparse
import einops
from functools import partial
import json
import os
import os.path as osp
import torch
import torch.nn as nn
from tqdm import tqdm
import traceback

import UtilsBase
from UtilsBase import to_json_encoding, from_json_encoding  # Use to encode/decode tensors in test cases
from Utils import twrite                                    # Useful for debugging

############ This *should* work automatically, but if not you can configure it #######
def get_terminal_columns(default=200):
    """Returns the number of columns in the terminal, or [default] if this does not
    work properly.
    """
    try:
        columns = os.get_terminal_size().columns
    except OSError:
        columns = default
    return columns
torch.set_printoptions(profile="full")
torch.set_printoptions(linewidth=get_terminal_columns())
######################################################################################
######################################################################################
######################################################################################

def output_ids_to_masks(*, output_ids, prompt_input_ids, pad_token_id, **kwargs):
    """Returns an dictionary with 'attention_mask', 'completion_mask', and
    'prompt_mask' keys, respectively mapping to int-valued (BS C)xT binary mask
    tensors [attention_mask], [completion_mask], and [prompt_mask], defined by:

    1. In [attention_mask], the ij index is 1 if the ij token of [output_ids] is a
        non-pad token and 0 if it is a pad token
    2. In [prompt_mask], the ij index is 1 if the ij token of [output_ids] is part of
        a "prompt region" and 0 otherwise
    3. In [completion_mask], the ij index is 1 if the ij token of [output_ids] is part
        of a "completion region" and 0 otherwise

    The PROMPT REGION of each sequence in [output_ids] is a contiguous span of tokens
    starting with the first non-pad token in the sequence and ending with the last
    token in the corresponding prompt in [prompt_input_ids].
    
    The COMPLETION REGION of each sequence in [output_ids] is simply all the tokens
    that come after its prompt region. It may contain pad tokens.

    Example:
    [PAD PAD    NON-PAD PAD NON-PAD    PAD PAD NON-PAD NON-PAD PAD PAD PAD]
    --------    -- prompt region --    --- completion region -------------

    HINTS: (1) Use torch.argmax() to find the start index of each prompt region:
        pytorch.org/docs/stable/generated/torch.argmax.html. (2) Infer the number of
        completions per prompt from the shapes of [output_ids] and [prompt_input_ids].
    ----------------------------------------------------------------------------------
    
    Args:
    output_ids          -- (BS C)xT tensor of token IDs returned by applying
                            model.generate() to [prompt_input_ids] together with other
                            arguments.

                            The first C sequences along the batch dimension contain
                            the C completions for the first prompt (first row of
                            [prompt_input_ids]), the next C sequences contain the C
                            completions for the second prompt, and so on.

    prompt_input_ids    -- BSxS tensor of token IDs used as input to model.generate()
    pad_token_id        -- token ID used for padding
    **kwargs            -- unused
    """
    raise NotImplementedError("Implement me!")


######################################################################################
# DO NOT IMPLEMENT THESE! Instead, implement output_ids_to_masks() instead.
# Correct implementations for them share some code, and they all have the same test
# cases. It might be easier to debug each component output separately though, so
# separate functions are provided.
######################################################################################
def output_ids_to_prompt_mask(*, output_ids, prompt_input_ids, pad_token_id, **kwargs):
    """Returns the prompt mask as defined in output_ids_to_masks()."""
    return vars(output_ids_to_masks(output_ids=output_ids,
        prompt_input_ids=prompt_input_ids,
        pad_token_id=pad_token_id,
        **kwargs)).get("prompt_mask", None)
def output_ids_to_completion_mask(*, output_ids, prompt_input_ids, pad_token_id, **kwargs):
    """Returns the completion mask as defined in output_ids_to_masks()."""
    return vars(output_ids_to_masks(output_ids=output_ids,
        prompt_input_ids=prompt_input_ids,
        pad_token_id=pad_token_id,
        **kwargs)).get("completion_mask", None)
def output_ids_to_attention_mask(*, output_ids, prompt_input_ids, pad_token_id, **kwargs):
    """Returns the attention mask as defined in output_ids_to_masks()."""
    return vars(output_ids_to_masks(output_ids=output_ids,
        prompt_input_ids=prompt_input_ids,
        pad_token_id=pad_token_id,
        **kwargs)).get("attention_mask", None)
######################################################################################
######################################################################################
######################################################################################

def get_per_token_logprobs(*, logits, token_ids, **kwargs):
    """Returns a BSx(T-1) tensor of log-probabilities, whose ij element is the
    log-probability of the j+1th token in the ith sequence of [token_ids], given all
    the preceeding tokens 0...j (inclusive) in the ith sequence of [token_ids].

    HINTS:
    1. Understanding logits: singlestore.com/blog/a-guide-to-softmax-activation-function
    2. torch.gather() is your friend: pytorch.org/docs/stable/generated/torch.gather.html
    3. Prefer log-softmax() over log(softmax()) for numerical stability

    Args:
    logits      -- BSxTxV tensor of logits output by an LM with vocab size V given
                    [token_ids] as input
    token_ids   -- BSxT tensor of token IDs
    **kwargs    -- unused
    """
    raise NotImplementedError("Implement me!")

def grpo_rl_objective(*, logprobs_model, logprobs_pi_old, advantages, grpo_clip=0.2, **kwargs):
    """Returns a BSxL tensor where the ij element is the GRPO policy objective.

    HINT: This is (equation 3) in the original GRPO paper (arxiv.org/pdf/2402.03300),
        but with two changes (1) the KL term is not computed here, and (2) it is
        defined over batches log-probabilities and advantages that don't necessarily
        correspond to a single group (ie. those from one prompt) of model outputs.

    Args:
    Suppose that originally there was a BSx(L+1) tensor of tokens fed to LMs pi_theta
    and pi_old. The ith sequences in [logprobs_model] and [logprobs_pi_old] contain
    conditional log-probabilities for the (zero-indexed) tokens 1...L in the ith
    sequence of this original tensor. Moreover, the ith element of [advantages]
    contains the advantage computed for completion contained within the ith sequence
    of this original tensor.

    logprobs_model  -- BSxL tensor of log-probabilities from the model pi_theta
    logprobs_pi_old -- BSxL tensor of log-probabilities from the old policy pi_old
    advantages      -- BS-length tensor of advantages
    grpo_clip       -- clipping parameter for GRPO
    **kwargs        -- unused
    """
    raise NotImplementedError("Implement me!")

def gpro_kl_loss(*, logprobs_model, logprobs_pi_ref, **kwargs):
    """Returns a BSxT tensor where the ij element is the modified KL loss for jth
    index of the ith element in the batch. Hint: equation (4) in the GRPO paper
    (arxiv.org/pdf/2402.03300.pdf).

    Args:
    Suppose that originally there was a BSx(T+1) tensor of tokens fed to LMs pi_theta
    and pi_ref. The ith sequences in [logprobs_model] and [logprobs_pi_ref] contain
    conditional log-probabilities for the (zero-indexed) tokens 1...T in the ith
    sequence of this original tensor.

    logprobs_model  -- BSxT tensor of log-probabilities from the model pi_theta
    logprobs_pi_ref -- BSxT tensor of log-probabilities from the model pi_ref
    **kwargs        -- unused
    """
    raise NotImplementedError("Implement me!")

def rewards_to_advantages(*, rewards, grpo_completions, **kwargs):
    """Returns a (BS C)-length tensor of advantages computed from [rewards] using
    GRPO's method.

    HINT: Add 1e-4 to the standard deviation for numerical stability.

    Args:
    rewards             --(BS C)-length tensor of rewards
    grpo_completions    -- number of completions per prompt
    """
    raise NotImplementedError("Implement me!")

@torch.no_grad()
def test_fn(*, fn, fn_args, fn_kwargs, fn_expected_output, device="cpu", verbose=False):
    """Tests [fn] with the given arguments and returns the output."""
    try:
        output = fn(*fn_args, device=device, **fn_kwargs)
    except Exception as e:
        tb = traceback.TracebackException.from_exception(e)
        error_trace = "".join(tb.format())
        tqdm.write("Function raised exception during evaluation of test inputs. Traceback:")
        tqdm.write(error_trace)
        return argparse.Namespace(
            passed=False,
            expected_output=fn_expected_output,
            actual_output=None)

    try:
        passed = torch.allclose(output, fn_expected_output)
    except Exception as e:
        passed = False
        tb = traceback.TracebackException.from_exception(e)
        error_trace = "".join(tb.format())
        tqdm.write("Function output could not be compared to expected output. Traceback:")
        tqdm.write(error_trace)
    return argparse.Namespace(
        passed=passed,
        expected_output=fn_expected_output,
        actual_output=output)

from Utils import twrite
if __name__ == "__main__":
    possible_functions = ["get_per_token_logprobs",
        "grpo_rl_objective", "gpro_kl_loss",
        "rewards_to_advantages",

        # These are really testing just the output_ids_to_masks() function, but
        # it may be easier to work on its components separately.
        "output_ids_to_attention_mask",
        "output_ids_to_completion_mask",
        "output_ids_to_prompt_mask"]
    P = argparse.ArgumentParser()
    P.add_argument("--fns_to_test", choices=possible_functions, nargs="+",
        default=possible_functions,
        help="List of functions to test.")
    P.add_argument("--device", type=str, default="cpu",
        help="Device to use for testing.")

    # Two options: (1) test_case_data.pt: uses torch.load() to get the test case data.
    # I only 99.999% trust this to work on arbitrary systems. (2) test_case_data.json:
    # custom (de)serialization functions in UtilsBase.py to reconstruct tensor data.
    P.add_argument("--test_case_data_path", type=str,
        default=osp.join(osp.dirname(__file__), "test_case_data.pt"),
        help="Path to the test case data")
    P.add_argument("--seed", type=int, default=42,
        help="Random seed for testing case generation.")
    P.add_argument("-v", "--verbose", action="store_true",
        help="Whether to print verbose output during testing.")
    P.add_argument("-vv", "--verbose2", action="store_true",
        help="Whether to print extra verbose output during testing.")
    args = P.parse_args()
    args.verbose = args.verbose or args.verbose2

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    
    if args.test_case_data_path.endswith(".pt"):
        tqdm.write(f"Loading test case data from PT file {args.test_case_data_path}...")
        fname2test_data = torch.load(args.test_case_data_path)
    elif args.test_case_data_path.endswith(".json"):
        tqdm.write(f"Loading test case data from JSON file {args.test_case_data_path}...")
        with open(args.test_case_data_path, "r+") as f:
            fname2test_data = json.load(f)
            fname2test_data = UtilsBase.from_json_encoding(fname2test_data)
    else:
        raise ValueError(f"Unknown test_case_data_path extension for {args.test_case_data_path}")


    failed_tests, total_tests = 0, 0
    for fn_name in args.fns_to_test:
        test_cases = fname2test_data[fn_name]
        fn = globals()[fn_name]

        print(f"Function={fn_name}: Running {len(test_cases)} test cases...")
        
        for idx,test_case in enumerate(test_cases):
            total_tests += 1
            fn_args = test_case["fn_args"]
            fn_kwargs = test_case["fn_kwargs"]
            fn_expected_output = test_case["fn_expected_output"]

            fn_result = test_fn(fn=fn, fn_args=fn_args, fn_kwargs=fn_kwargs, fn_expected_output=fn_expected_output, device=device, verbose=args.verbose)
            failed_tests += int(not fn_result.passed)
            test_result_str = "PASSED" if fn_result.passed else "FAILED"
            
            if not args.verbose and not args.verbose2:
                tqdm.write(f"\tFunction={fn_name} Test case {idx+1}/{len(test_cases)} {test_result_str}")
            elif args.verbose and not args.verbose2:
                tqdm.write(f"\t=======================================================")
                tqdm.write(f"\tFunction={fn_name} Test case {idx+1}/{len(test_cases)} {test_result_str}")
                tqdm.write(f"\tExpected output:\n{fn_result.expected_output}")
                tqdm.write(f"\tGot output:\n{fn_result.actual_output}")
            elif args.verbose2:
                tqdm.write(f"\t=======================================================")
                tqdm.write(f"\tFunction={fn_name} Test case {idx+1}/{len(test_cases)} {test_result_str}")
                tqdm.write(f"\tFunction positional arguments (total={len(fn_args)}):")
                for arg_idx,a in enumerate(fn_args):
                    print_newline = "\n" if isinstance(a, torch.Tensor) and a.ndim > 1 else ""
                    tqdm.write(f"Arg {arg_idx}: {print_newline}{a}")
                tqdm.write(f"\tFunction keyword arguments (total={len(fn_kwargs)}):")
                for k,v in fn_kwargs.items():
                    print_newline = "\n" if isinstance(v, torch.Tensor) and v.ndim > 1 else ""
                    tqdm.write(f"{k}={print_newline}{v}")
                tqdm.write(f"\t=== Expected output:\n{fn_result.expected_output}")
                tqdm.write(f"\t=== Received output:\n{fn_result.actual_output}")

    tqdm.write(f"Testing complete: {total_tests-failed_tests}/{total_tests} tests passed.")



