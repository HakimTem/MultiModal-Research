from torch.utils.data import Dataset
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import WhisperProcessor

from IPython.display import display
from torch.utils.data import DataLoader, Subset
from bitsandbytes.optim import Adam8bit
import math
from einops import rearrange
from tqdm import tqdm

from datasets import config
config.DOWNLOADER_TIMEOUT = 36000

import fsspec
fsspec.config.conf['http_timeout'] = 36000 

import numpy as np

from CombinedGMC.combined_gmc import CombinedGMCModel

# End of sentence token
ANSWER_EOS = "<|endoftext|>"

# Number of tokens used to represent each image.
IMG_TOKENS = 729

# Number of tokens used to represent each audio clip.
AUDIO_TOKENS = 1500

# Method for collecting features for a given batch of samples
def collate_fn(batch):
    # preprocess the images
    images = [sample['image'] for sample in batch]
    images = [combined_model.moondream.vision_encoder.preprocess(image) for image in images]

    audios = np.random.randn(16000 * 10)
    audios = combined_model.audio_processor(audios, sampling_rate=16000, return_tensors="pt", padding="max_length")["input_features"]

    # labels accumulator
    labels_acc = []
    tokens_acc = []

    # for every sample in the batch then tokenize the question and answer
    for sample in batch:
        # start with the bos token
        toks = [combined_model.tokenizer.bos_token_id]

        # add the image and audio tokens
        labs = [-100] * (IMG_TOKENS + 1)
        labs = labs + ([-100] * (AUDIO_TOKENS))

        # add the question and answer tokens
        for qa in sample['qa']:
            q_t = combined_model.tokenizer(
                f"\n\nQuestion: {qa['question']}\n\nAnswer:",
                add_special_tokens=False
            ).input_ids
            toks.extend(q_t)
            labs.extend([-100] * len(q_t))

            a_t = combined_model.tokenizer(
                f" {qa['answer']}{ANSWER_EOS}",
                add_special_tokens=False
            ).input_ids
            toks.extend(a_t)
            labs.extend(a_t)

        tokens_acc.append(toks)
        labels_acc.append(labs)

    max_len = -1
    for labels in labels_acc:
        max_len = max(max_len, len(labels))

    attn_mask_acc = []

    for i in range(len(batch)):
        len_i = len(labels_acc[i])
        pad_i = max_len - len_i

        labels_acc[i].extend([-100] * pad_i)
        tokens_acc[i].extend([combined_model.tokenizer.eos_token_id] * pad_i)
        attn_mask_acc.append([1] * len_i + [0] * pad_i)

    audios = audios.to(DEVICE)

    return (
        images,
        audios,
        torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
        torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
        torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
    )

# def compute_loss(batch):
#     images, tokens, labels, attn_mask = batch

#     tokens = tokens.to(DEVICE)
#     labels = labels.to(DEVICE)
#     attn_mask = attn_mask.to(DEVICE)

#     # with torch.no_grad():
#     img_embs = moondream.vision_encoder(images)

#     tok_embs = moondream.text_model.get_input_embeddings()(tokens)
#     inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)

#     outputs = moondream.text_model(
#         inputs_embeds=inputs_embeds,
#         labels=labels,
#         attention_mask=attn_mask,
#         output_hidden_states=True
#     )

#     print(outputs.hidden_states[-1].shape)

#     return outputs.loss

def gmc_loss(img_embs, tok_embs, batch_size, temperature):
    pass

def lr_schedule(step, max_steps):
    x = step / max_steps
    if x < 0.1:
        return 0.1 * LR + 0.9 * LR * x / 0.1
    else:
        return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2

class CaptchaDataset(Dataset):
    def __init__(self, split='train'):
        self.data = load_dataset("google/docci", trust_remote_code=True, storage_options={"timeout": 36000})[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "image": sample["image"], # Should be a PIL image
            "qa": [
                {
                    "question": "Describe this image.",
                    "answer": sample["description"],
                }
            ]
        }

if __name__ == "__main__":

    datasets = {
        "train": CaptchaDataset("train"),
        "test": CaptchaDataset("test"),
    }


    DEVICE = "cuda"
    DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16 # CPU doesn't support float16
    MD_REVISION = "2024-07-23"

    combined_model = CombinedGMCModel(0.1)
    # A = [np.random.randn(16000 * 5), np.random.randn(16000 * 5)]
    # audios = combined_model.audio_processor(A, sampling_rate=16000, return_tensors="pt", padding="max_length")
    # print(audios["input_features"].shape)

    # print(combined_model.audio_model.encoder(audios["input_features"]).last_hidden_state)

    # tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)
    # moondream = AutoModelForCausalLM.from_pretrained(
    #     "vikhyatk/moondream2", revision=MD_REVISION, trust_remote_code=True,
    #     attn_implementation=None,
    #     torch_dtype=DTYPE, device_map={"": DEVICE}
    # )

    # Test the model on one sample
    # sample = datasets['train'][0]
    # image = sample['image']
    # image_embeds = moondream.vision_encoder(image)


    # # for qa in sample['qa']:
    # #     print('Question:', qa['question'])
    # #     print('Ground Truth:', qa['answer'])
    # #     print('Moondream:', moondream.answer_question(
    # #         moondream.encode_image(sample['image']),
    # #         qa['question'],
    # #         tokenizer=tokenizer,
    # #     ))


    # Finetune the model
    # Number of times to repeat the training dataset. Increasing this may cause the model to overfit or
    # lose generalization due to catastrophic forgetting. Decreasing it may cause the model to underfit.
    EPOCHS = 1

    # Number of samples to process in each batch. Set this to the highest value that doesn't cause an
    # out-of-memory error. Decrease it if you're running out of memory.
    BATCH_SIZE = 1

    # Number of batches to process before updating the model. You can use this to simulate a higher batch
    # size than your GPU can handle. Set this to 1 to disable gradient accumulation.
    GRAD_ACCUM_STEPS = 2

    # Learning rate for the Adam optimizer. Needs to be tuned on a case-by-case basis. As a general rule
    # of thumb, increase it by 1.4 times each time you double the effective batch size.
    #
    # Source: https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    #
    # Note that we linearly warm the learning rate up from 0.1 * LR to LR over the first 10% of the
    # training run, and then decay it back to 0.1 * LR over the last 90% of the training run using a
    # cosine schedule.
    LR = 1e-5

    # Whether to use Weights and Biases for logging training metrics.
    USE_WANDB = False


    subset = Subset(datasets['train'], range(1))
    dataloaders = {
        "train": DataLoader(
            subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
        )
    }

    combined_model.train()
    combined_model.moondream.text_model.gradient_checkpointing_enable()

    total_steps = EPOCHS * len(dataloaders["train"]) // GRAD_ACCUM_STEPS
    optimizer = Adam8bit(
        [
            {"params": combined_model.parameters()},
        ],
        lr=LR * 0.1,
        betas=(0.9, 0.95),
        eps=1e-6
    )

    # if USE_WANDB:
    #     import wandb
    #     wandb.init(
    #         project="moondream-ft",
    #         config={
    #             "EPOCHS": EPOCHS,
    #             "BATCH_SIZE": BATCH_SIZE,
    #             "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
    #             "LR": LR,
    #         }
    #     )

    i = 0
    for epoch in range(EPOCHS):
        for batch in tqdm(dataloaders["train"], desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            i += 1

            # loss = compute_loss(batch)
            loss = combined_model(batch)
            loss.backward()

            if i % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                lr = lr_schedule(i / GRAD_ACCUM_STEPS, total_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            # if USE_WANDB:
            #     wandb.log({
            #         "loss/train": loss.item(),
            #         "lr": optimizer.param_groups[0]['lr']
            #     })

    # if USE_WANDB:
    #     wandb.finish()
    # moondream.save_pretrained("checkpoints/moondream+gmc-ft")


    # Load and use the fine-tuned model
    # moondream = AutoModelForCausalLM.from_pretrained("checkpoints/moondream-ft", use_safetensors=True, trust_remote_code=True)
    # moondream.eval()

    # subset = Subset(datasets['test'], range(5))

    # for i, sample in enumerate(subset):
    #     print(i)
        
    #     md_answer = moondream.answer_question(
    #         moondream.encode_image(sample['image']),
    #         sample['qa'][0]['question'],
    #         tokenizer=tokenizer, 
    #         early_stopping=True
    #     )

    #     if i < 3:
    #         display(sample['image'])
    #         print('Question:', sample['qa'][0]['question'])
    #         print('Ground Truth:', sample['qa'][0]['answer'])
    #         print('Moondream:', md_answer)
    #     else:
    #         break

    