import logging

import hydra
import pandas as pd
import torch
from PIL import Image
from rich import traceback
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from tqdm import tqdm
from dataset import Multi30kDataset, COCO35Dataset, XM3600Dataset
from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)

logging.basicConfig(
    handlers=[RichHandler(rich_tracebacks=True)],
)

traceback.install()


def prepare_batch(batch, processor, prefix):
    images, captions, image_paths = zip(*batch)

    if isinstance(processor, PaliGemmaProcessor):
        prefix = prefix.strip()
        batch = processor(
            images=images,
            text=[prefix] * len(captions),
            # suffix=captions,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )

        captions = [f"{prefix}{caption}" for caption in captions]
    else:
        raise ValueError(f"Processor {processor} not implemented.")
        # captions = [f"{prefix}{caption}" for caption in captions]
        # batch = processor(
        #     captions, return_tensors="pt", padding="longest", add_special_tokens=True
        # )
    batch.update({"image_paths": image_paths, "captions": captions})
    return batch


def load_model(cfg):
    kwargs = {}
    if cfg.quant:
        kwargs.update(
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": False,
                "bnb_4bit_compute_dtype": torch.bfloat16,
            }
        )
    if cfg.name in ["paligemma"]:  # "mblip-bloomz", "mblip-mt0"]:
        model = PaliGemmaForConditionalGeneration.from_pretrained(cfg.path, **kwargs)

        processor = AutoProcessor.from_pretrained(
            cfg.path,
            padding_side="right",
        )
        processor.tokenizer.padding_side = "right"
    elif cfg.name in ["gemma-2b", "ft-pali"]:
        model = AutoModelForCausalLM.from_pretrained(cfg.path)
        processor = AutoTokenizer.from_pretrained(cfg.tok_path, padding_side="right")
        processor.add_eos_token = True
        # processor.add_special_tokens({"eos_token": "\n"})
    else:
        raise ValueError(f"Model {cfg.name} not implemented.")
    return model, processor


def get_data(cfg, processor, tokenizer):
    # compute prefix length
    prefix = f"caption {cfg.lang}\n"
    prefix_len = len(tokenizer(prefix)["input_ids"]) - 1

    if cfg.dataset.name == "test":
        # images = [Image.open("data/multi30k/images/1001465944.jpg")]
        images = [Image.open("data/multi30k/images/227689211.jpg")]
        images += [Image.open(cfg.dataset.path)] * 4 + [Image.open("test2.jpg")]
        captions = [
            # "A woman with a large purse is walking by a gate.",
            "A police officer in his uniform wearing an ear piece.",
            "A test sentence.",
            "A bicycle replica with a clock as the front wheel.",
            "A cat is laying on top of a dryer.",
            "A test sentence with indubitably obscure verbage.",
            "Two dogs and a cat.",
            "Two cats and a dog.",
            "the the the the the." "A bicycle replica with a clock as the front wheel.",
        ]
        image_paths = (
            ["data/multi30k/1001465944.jpg"] + [cfg.dataset.path] * 4 + ["test2.jpg"]
        )
        data = [
            prepare_batch(
                zip(*(images, captions, image_paths)), processor, prefix=prefix
            )
        ]
        return data, prefix_len
    elif cfg.dataset.name == "xm3600":
        dataset_class = XM3600Dataset
    elif cfg.dataset.name == "coco35":
        dataset_class = COCO35Dataset
    elif cfg.dataset.name == "multi30k":
        dataset_class = Multi30kDataset
    else:
        raise ValueError(f"Dataset {cfg.dataset.name} not implemented.")
    data = DataLoader(
        dataset_class(
            cfg.dataset.path,
            cfg.dataset.split,
            cfg.lang,
        ),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=lambda b: prepare_batch(b, processor, prefix=prefix),
    )
    return data, prefix_len


@hydra.main(config_path="../config", config_name="config", version_base="1.2")
def main(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, processor = load_model(cfg.model)
    processor.padding_side = "left"  # for generation
    model.eval().to(device)

    if hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
    else:
        tokenizer = processor

    data, prefix_len = get_data(cfg, processor, tokenizer)
    full_results = []

    # pdb.set_trace()
    for batch in tqdm(data):
        inputs = {
            k: batch[k].to(device)
            for k in [
                "pixel_values",
                "token_type_ids",
                "labels",
                "input_ids",
                "attention_mask",
            ]
        }
        with torch.no_grad():
            outputs = model.generate(**inputs, use_cache=True)
            predictions = tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            )
            print(predictions)
            results = {
                "prediction": predictions,
                "image_id": batch["image_paths"],
                "caption": batch["captions"],
                "lang": cfg.lang,
            }
        full_results.append(pd.DataFrame(results))

    full_results = pd.concat(full_results, ignore_index=True)
    hydra_output = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out_file = f"{hydra_output}/../../../{cfg.out_file}"
    # add a date and time column
    full_results["date"] = pd.to_datetime("today").strftime("%Y-%m-%d-%H-%M")
    with open(out_file, "w") as f:
        full_results.to_csv(f, index=False)
    # os.symlink(
    #     f"{hydra_output}/{out_file.name}", f""
    # )


if __name__ == "__main__":
    main()
