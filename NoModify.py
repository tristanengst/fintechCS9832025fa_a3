import argparse
from collections import defaultdict
import copy
import datasets as hf_datasets
import einops
import os
import os.path as osp
import json
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import Utils
from Utils import twrite
import UtilsBase

def get_system_prompt():
    prompt_base = f"Your task is rewriting headlines to be perceived with a new target sentiment by changing tone, adding stylistic flair, or adding related made-up details that support the target sentiment but do not contradict the original. Otherwise the rewritten headline closely resembles the original and includes the same factual information. You output only the text of the rewritten headline. Never state target sentiment directly."
    return prompt_base

def get_data_from_file(fname):
    """Returns the list of headlines within the dataset given by JSON file [fname]."""
    with open(fname, "r+") as f:
        headline2sentiment = json.load(f)
    return [dict(headline=h, sentiment=s) for h,s in headline2sentiment.items() if len([hh for hh in h.split() if len(hh)>4]) >= 8 and len(h.split()) <= 24]
    
class CollateFn:
    """Class implementing a collate function for DataLoader.

    Args:
    model_tokenizer     -- tokenizer for the policy model
    finbert_tokenizer    -- tokenizer for the reward model
    """
    def __init__(self, *,  model_tokenizer):
        self.model_tokenizer = model_tokenizer

    def __call__(self, batch):
        """Returns a collated version of [batch] for use in DataLoader. Any keys whose
        values are tensor-valued are collated with the appropriate tokenizer's
        DataCollatorWithPadding. This will turn a BS-length list of
        possibly-different-length tensors containing input_ids and attention_mask for
        prompts, headlines, and sentiments into properly padded BSxT tensors.

        Args:
        batch       -- list of examples, each of which is a dict with the keys
                        - prompt_input_ids
                        - prompt_attention_mask
                        as well as possibly other keys (eg. the original headline, sentiment, and prompt strings)
        """
        headlines = [hs["headline"] for hs in batch]
        sentiments = [hs["sentiment"] for hs in batch]
        prompts = [hs["prompt"] for hs in batch]

        batch_prompt = self.model_tokenizer(prompts, truncation=True, padding=True, return_tensors="pt")
        batch_prompt = {f"prompt_{k}": v for k,v in batch_prompt.items()}
        return batch_prompt | dict(sentiment=sentiments, headline=headlines, prompt=prompts) 

def get_data(*, data_path, model_tokenizer, args, write_data=False, train=True):
    """Returns training and validation DataLoaders given [args].
    
    Args:
    model_tokenizer     -- tokenizer for the policy model
    train               -- whether the returned data is for training or validation/testing
    args                -- Namespace with arguments
    """
    def write_data_to_file(data, fname):
        """Writes [data] to JSON file [fname]."""
        headline2sentiment = {d["headline"]: d["sentiment"] for d in data}
        with open(fname, "w+") as f:
            json.dump(headline2sentiment, f, indent=4)
        twrite(f"Wrote {len(data)} examples to {fname}")

    def dataset_to_sampler_loader(data, *, batch_size, shuffle=False, drop_last=False,
        seed=None, num_workers=8, persistent_workers=True,
        pin_memory=False, multiprocessing_context="fork", collate_fn=None):
        """Returns a (sampler, loader) tuple for [data]."""
        sampler = DistributedSampler(data, num_replicas=1, rank=0, seed=seed, shuffle=shuffle)

        persistent_workers = persistent_workers if num_workers > 0 else False
        return sampler, DataLoader(data, batch_size=batch_size,
            sampler=sampler, collate_fn=collate_fn, drop_last=drop_last,
            num_workers=num_workers, persistent_workers=args.persistent_workers,
            pin_memory=pin_memory,
            multiprocessing_context=multiprocessing_context)

    def make_prompt(headline, sentiment):
        s = f"Target: {sentiment}. Headline: {headline}"
        messages = [dict(role="system", content=get_system_prompt())]
        messages += [dict(role="user", content=s)]
        return model_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Load a dataset to build off of
    sentiments = ["negative", "positive"]
    if osp.exists(data_path):
        headlines_sentiments = get_data_from_file(fname=args.data_tr)
        headlines = [hs["headline"] for hs in headlines_sentiments]
    else:
        raise NotImplementedError()
    
    # The actual dataset we care about is a list of prompts, where each prompt is a
    # (headline, sentiment, prompt) triple. The LM we're training needs just the
    # prompt, while the reward model needs to know the headline and sentiment as well.
    data = [dict(headline=h, sentiment=s) for h in headlines for s in sentiments]
    data = data[:args.data_max_ex] if args.data_max_ex else data
    data = [d | dict(prompt=make_prompt(d["headline"], d["sentiment"])) for d in data]    
    data = hf_datasets.Dataset.from_list(data)

    # Create the collate funtion. Essentially, it just needs to pad the input_ids and
    # attention_mask for headlines, sentiments, and prompts individually to have the
    # same sequence length within a batch.
    
    collate_fn = CollateFn(model_tokenizer=model_tokenizer)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"          # Apparently safe, though we should probably not be forking stuff...
    dataloader_kwargs = dict(num_workers=args.num_workers,  # Number of subprocesses to use for data loading. This is the first thing to tune for performance
        pin_memory=True,                                    # Probably slightly faster, consider turning off if training is unstable
        persistent_workers=True,                            # Keep workers alive between epochs. Slightly faster, but might be less stable
        multiprocessing_context=args.hw_mp_ctx,             # 'fork' is generally fastest but can be unstable especially when run for longer
        collate_fn=collate_fn,
        drop_last=train,
        shuffle=train)
    
    sampler, loader = dataset_to_sampler_loader(data, batch_size=args.bs,
        seed=args.seed, **dataloader_kwargs)
    return sampler, loader

class SequenceRewardModel(nn.Module):
    """Neural network used to compute sequence-level rewards."""
    def __init__(self, args=None):
        super(SequenceRewardModel, self).__init__() 
        self.args = copy.deepcopy(args)
        self.device = Utils.get_device(self.args)

        # Setup FinBERT for sentiment analysis
        self.sentiment2idx = {"neutral": 0, "positive": 1, "negative": 2} # Sentiment-to-label map used by FinBERT
        self.finbert_tokenizer_kwargs = dict(padding=True, truncation=True, padding_side="right", return_tensors="pt")
        self.finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-pretrain",
            use_fast=True, model_max_length=512, **self.finbert_tokenizer_kwargs)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone",
            device_map=None if torch.cuda.is_available() else "cpu",)
        self.sentiment_model.eval()
        for param in self.sentiment_model.parameters():
            param.requires_grad = False

        # Setup the content embedding model
        self.content_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", use_fast=True, model_max_length=512)
        self.content_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.content_model.eval()
        for param in self.content_model.parameters():
            param.requires_grad = False
        embedded_system_prompt = self.content_model.encode(get_system_prompt(),
            convert_to_tensor=True, normalize_embeddings=True).to(self.device)
        embedded_system_prompt = einops.rearrange(embedded_system_prompt, "d -> 1 d")
        bad_strings = ["New headline", "Target sentiment", "Target positive",
            "Target negative", "Target neutral",]
        embedded_system_prompt2 = self.content_model.encode(bad_strings,
            convert_to_tensor=True, normalize_embeddings=True).to(self.device)
        self.embedded_system_prompt = torch.cat([embedded_system_prompt,
            embedded_system_prompt2], dim=0)
        

    @torch.no_grad()
    def forward_sentiment_logits(self, *, input_ids, attention_mask, **kwargs):
        """Returns the BSx3 tensor of sentiment logits for [input_ids] processed with
        [attention_mask].

        Args:
        input_ids       -- BSxT tensor of token IDs
        attention_mask  -- BSxT tensor giving the attention mask
        """
        outputs = self.sentiment_model(input_ids=input_ids, attention_mask=attention_mask,
            return_dict=True)
        return outputs.logits
        
    @torch.no_grad()
    def forward(self, *, completions, sentiments, headlines):
        """Returns a (BS C)-length vector of sequence-level rewards for the completions
        responses [completions], given the target [sentiments] and original [headlines].

        BS is the batch size, while C is the number of completions candidates for each
        of the BS prompts. Alternatively, if we've already 'expanded' the headlines
        to be of length (BS C), then C=1 effectively and this will work anyways.

        Args:
        completions   -- (BS C)xL tensor of token IDs for candidate headlines. Each
                        group of [C]  consecutive entries correspond to the [C]
                        completions for a single prompt in the batch (and thus have
                        the same target sentiment and original headline).
        sentiments  -- BS- or (BS C)-length list of target sentiments
        headlines   -- BS- or (BS C)-length list of original headlines
        headlines_tokenized -- dict with 'input_ids' and 'attention_mask' tensors for the original headlines
        """
        assert len(sentiments) == len(headlines), "Sentiments and headlines must have the same batch size!"
        bs_c = len(completions)
        bs = len(sentiments)
        c = bs_c // bs
        # assert c == 1 or (c % self.args.cbs == 0), f"Got c={c} cbs={self.args.cbs} bs_c={bs_c} bs={bs}"

        # Indices of completions with target sentiments.
        # S = ["negative", "neutral", "positive"]
        # S += [s.upper() for s in S]


        # Get embeddings for [completions] and [headlines] from the content model. These
        # are used to compute their similarity.
        completions_embeds = self.content_model.encode(completions, convert_to_tensor=True, normalize_embeddings=True)
        completions_embeds = einops.rearrange(completions_embeds, "(bs c) d -> bs c d", bs=bs, c=c)
        headline_embeds = self.content_model.encode(headlines, convert_to_tensor=True, normalize_embeddings=True)
        content_sims = einops.einsum(completions_embeds, headline_embeds, "bs c d, bs d -> bs c") # Dot-product over last d-dimension
        content_sims = einops.rearrange(content_sims, "bs c -> (bs c)")
        content_reward = torch.exp(content_sims) - 0.9

        # Compute the similarity between [completions] and the system prompt to penalize
        bad_content_sims = einops.einsum(completions_embeds, self.embedded_system_prompt, "bs c d, b d -> bs c b")
        bad_content_sims = einops.reduce(bad_content_sims, "bs c b -> bs c", "sum")
        bad_content_sims = einops.rearrange(bad_content_sims, "bs c -> (bs c)")
        bad_content_reward = torch.exp(bad_content_sims) - 0.9

        # Get the sentiment predictions for [completions]
        answers_sentiment_ids = self.finbert_tokenizer(completions, **self.finbert_tokenizer_kwargs).to(self.device)
        sentiment_logits = self.forward_sentiment_logits(**answers_sentiment_ids)


        # Repeat-interleave [sentiments] to have shape (BS C) to extract the target
        # sentiments for each element of [completions]
        sentiment_idxs = torch.tensor([self.sentiment2idx[s] for s in sentiments], device=self.device)
        sentiment_idxs = einops.repeat(sentiment_idxs, "bs -> (bs c)", c=c)
        sentiment_pred = nn.functional.softmax(sentiment_logits, dim=-1)
        sentiment_pred = torch.gather(sentiment_pred, 1, einops.rearrange(sentiment_idxs, "bsc -> bsc 1"))
        sentiment_pred = einops.rearrange(sentiment_pred, "bsc 1 -> bsc")
        sentiment_reward = torch.exp(sentiment_pred) - 0.9

        # For each index in sentiment_idxs, check whether the corresponding completion
        # contains the target sentiment at all. We need a binary (BS C)-length mask
        # for this.
        sentiment_ids = torch.tensor([1527, 483, 591], device=self.device)  # neutral, positive, negative
        contains_pos = (answers_sentiment_ids.input_ids.int() == 483)
        contains_neg = (answers_sentiment_ids.input_ids.int() == 591)
        contains_neu = (answers_sentiment_ids.input_ids.int() == 1527)
        contains_sentiment_id = (contains_pos | contains_neg | contains_neu).sum(dim=-1)
        num_tokens_in_completion = answers_sentiment_ids.attention_mask.sum(dim=-1)
        num_tokens_in_completion = torch.clamp(torch.log(num_tokens_in_completion.float()), min=1)
        contains_sentiment_id = contains_sentiment_id.float() / num_tokens_in_completion

        # Per-sequence reward is product of content and sentiment rewards, minus the
        # penalty for bad content similar to the system prompt wherever it's positive
        total_reward = content_reward * sentiment_reward - torch.max(torch.zeros_like(bad_content_reward), bad_content_reward) - contains_sentiment_id
        return total_reward

