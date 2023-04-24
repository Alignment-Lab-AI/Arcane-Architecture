import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
#Processes the input sequence in segments.
#Concatenate memory vectors with segment tokens, as described by H˜0τ = [Hmemτ ◦ H0τ].
#Passes the combined memory and segment tokens through the Transformer model, as described by H¯Nτ = Transformer(H˜0τ).
#Updates the memory vectors after the forward pass, as described by Hmemτ+1 := H¯memτ.
#Passes the updated memory vectors to the next segment, as described by H˜0τ+1 = [Hmemτ+1 ◦ H0τ+1].
#new attention layer on decoder only?

class RecurrentMemoryTransformer(nn.Module):
    def __init__(self, model_name, num_memory_vectors):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.memory_vectors = nn.Parameter(torch.randn(num_memory_vectors, self.transformer.config.hidden_size))

    def forward(self, input_ids, segment_size):
        device = input_ids.device
        batch_size = input_ids.size(0)

        memory = self.memory_vectors.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        outputs = []

        for i in range(0, input_ids.size(1), segment_size):
            segment = input_ids[:, i:i + segment_size]
            segment_input = torch.cat([memory, segment], dim=1)
            segment_output = self.transformer(input_ids=segment_input).last_hidden_state

            updated_memory = segment_output[:, :memory.size(1), :]
            output = segment_output[:, memory.size(1):, :]
            memory = updated_memory

            outputs.append(output)

        return torch.cat(outputs, dim=1)

class CustomT5DecoderLayerWithSegment(nn.Module):
    def __init__(self, layer, cache_len):
        super().__init__()
        self.layer = layer
        self.cache_len = cache_len
        self.key_cache = None
        self.value_cache = None

    def forward(self, input, **kwargs):
        if self.key_cache is None:
            self.key_cache = input.new_zeros((input.size(0), self.cache_len, input.size(-1)))
            self.value_cache = input.new_zeros((input.size(0), self.cache_len, input.size(-1)))

        input_with_key_cache = torch.cat([self.key_cache, input], dim=1)
        input_with_value_cache = torch.cat([self.value_cache, input], dim=1)

        kwargs["cross_attention_hidden_states"] = input_with_key_cache
        kwargs["cross_attention_hidden_states_values"] = input_with_value_cache

        output = self.layer(input, **kwargs)
        self.key_cache = input_with_key_cache[:, -self.cache_len:]
        self.value_cache = input_with_value_cache[:, -self.cache_len:]

        return output


class CustomT5DecoderWithSegment(nn.Module):
    def __init__(self, decoder, cache_len):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomT5DecoderLayerWithSegment(layer, cache_len)
            for layer in decoder.block
        ])
        self.final_layer_norm = decoder.final_layer_norm

    def forward(self, input, **kwargs):
        for layer in self.layers:
            input = layer(input, **kwargs)
        return self.final_layer_norm(input)


class CombinedApproachT5Decoder(nn.Module):
    def __init__(self, model_name, num_memory_vectors, segment_size, cache_len):
        super().__init__()
        self.transformer = T5ForConditionalGeneration.from_pretrained(model_name)
        self.memory_vectors = nn.Parameter(torch.randn(num_memory_vectors, self.transformer.config.d_model))
        self.segment_size = segment_size
        self.cache_len = cache_len
        self.custom_decoder = CustomT5DecoderWithSegment(self.transformer.decoder, cache_len)

    def forward(self, input_ids, decoder_input_ids):
        device = input_ids.device
        batch_size = input_ids.size(0)

        memory = self.memory_vectors.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        outputs = []

        # Encoder
        encoder_output = self.transformer.get_encoder()(input_ids=input_ids).last_hidden_state

        # Decoder with segment-level processing and layer-level caching
        for i in range(0, decoder_input_ids.size(1), self.segment_size):
            segment = decoder_input_ids[:, i:i + self.segment_size]
            segment_input = torch.cat([memory, segment], dim=1)
            segment_output = self.custom_decoder(segment_input, encoder_hidden_states=encoder_output)

            updated_memory = segment_output[:, :memory.size(1), :]
            output = segment_output[:, memory.size(1):, :]
            memory = updated_memory

            outputs.append(output)

        return torch.cat(outputs, dim=1)


model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
num_memory_vectors = 5
segment_size = 10
cache_len = 50

model = CombinedApproachT5Decoder(model_name, num_memory_vectors, segment_size, cache_len)

input_text = "This is an example input."
decoder_input_text = "This is an example output."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
decoder_input_ids = tokenizer(decoder_input_text, return_tensors="pt").input_ids

output = model(input_ids, decoder_input_ids, segment_size)

print(output)