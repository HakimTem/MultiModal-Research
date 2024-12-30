import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import WhisperProcessor, WhisperModel

class CausalLMHead(nn.Module):
    """Causal Language Modeling head. Simplified version."""

    def __init__(self, hidden_dim, shared_dim):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, shared_dim)

    def forward(self, hidden_states):
        return self.linear(self.ln(hidden_states))

class CombinedGMCModel(nn.Module):
    def __init__(self, temperature):
        super(CombinedGMCModel, self).__init__()
        self.temperature = temperature

        # tokenizer+model for moondream joint classifier
        self.tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision="2024-07-23")
        self.moondream = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2", revision="2024-07-23", trust_remote_code=True,
            attn_implementation=None,
            torch_dtype=torch.float16, device_map={"": "cuda"}
        )

        self.audio_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.audio_model = WhisperModel.from_pretrained("openai/whisper-base").to("cuda")
        self.audio_projector = CausalLMHead(512, 2048).to("cuda")

    def forward(self, batch):
        batch_size = len(batch[0])
        images, audios, tokens, labels, attn_mask = batch

        tokens = tokens.to("cuda")
        labels = labels.to("cuda")
        attn_mask = attn_mask.to("cuda")

        with torch.enable_grad():
            img_embs = self.moondream.vision_encoder(images)

        # processed_audios = self.audio_processor(audios, sampling_rate=16000, return_tensors="pt")
        # print(audios.shape)
        audio_embeds = self.audio_model.encoder(audios).last_hidden_state
        audio_embeds = self.audio_projector(audio_embeds).half()


        tok_embs = self.moondream.text_model.get_input_embeddings()(tokens)
        inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, audio_embeds, tok_embs[:, 1:, :]), dim=1)

        joint_outputs = self.moondream.text_model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attn_mask,
            output_hidden_states=True
        )

        batch_representations = [img_embs[:, -1, :], audio_embeds[:, -1, :], joint_outputs.hidden_states[-1][:, -1, :]]

        joint_mod_loss_sum = 0
        for mod in range(len(batch_representations) - 1):
            # Negative pairs: everything that is not in the current joint-modality pair
            out_joint_mod = torch.cat(
                [batch_representations[-1], batch_representations[mod]], dim=0
            )
            # [2*B, 2*B]
            sim_matrix_joint_mod = torch.exp(
                torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / self.temperature
            )
            # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
            mask_joint_mod = (
                torch.ones_like(sim_matrix_joint_mod)
                - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
            ).bool()
            # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
            sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(
                mask_joint_mod
            ).view(2 * batch_size, -1)

            # Positive pairs: cosine loss joint-modality
            pos_sim_joint_mod = torch.exp(
                torch.sum(
                    batch_representations[-1] * batch_representations[mod], dim=-1
                )
                / self.temperature
            )
            # [2*B]
            pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)
            loss_joint_mod = -torch.log(
                pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1)
            )
            joint_mod_loss_sum += loss_joint_mod

        # Compute the GMC loss
        return torch.mean(joint_mod_loss_sum + joint_outputs.loss)
    