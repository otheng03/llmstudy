import tiktoken
import torch
from models import GPTModel, generate_text_simple, create_dataloader_v1

"""
Pretraining on unlableded data

This chapter covers:
- Computing the training and validation set losses to assess the quality of LLM-generated text during training
- Implementing a training function and pretraining the LLM
- Saving and loading model weights to continue training an LLM
- Loading pretrained weights from OpenAI

Weights
- In the context of LLMs and other deep learning models, weights refer to the trainable parameters that
  the learning process adjusts.
- These weights are also known as weight parameters or simple parameters.
"""

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,  # Reduced it to 256 in order to carry out the training on a standard laptop computer
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
tokenizer = tiktoken.get_encoding("gpt2")

# facilitate = to make (something) easier

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def execute_untrained_model():
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    start_context = "Every effort moves you"

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output text using an untrained model:\n", token_ids_to_text(token_ids, tokenizer))
execute_untrained_model()

"""
Backpropagation:
In order to maximize the softmax probability values corresponding to the target tokens, you can update the model weights.
Via the update process, the model outputs higher values for the respective token IDs we want to generate.
The weigh update is done via a process called backpropagation, a standard technique for training deep neural networks.
Backpropagation requires a loss function, which calculates the difference between the model's predicted output 
(here, the probabilities corresponding to the target token IDs) and the actual desired output.
This loss function measures how far off the model's predictions are from the target values.

Cross entropy loss:
It is a popular measure in machine learning and deep learning that measures the difference between two probability 
distributions--typically, the true distribution of labels and the predicted distribution from a model.

Calculating the loss:
1. Logits
2. Probabilities
3. Target probabilities
4. Log probabilities
5. Average log probability
6. Negative average log probability (=the loss we want to compute)
"""
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

"""
A typical training loop for training deep neural networks in PyTorch:
1. Iterate over training epochs
  2. Iterate over batches in each training epoch
    3. Reset loss gradients from previous batch iteration
    4. Calculate loss on current batch
    5. Backward pass to calculate loss gradients
    6. Update model weights using loss gradients
    7. Print training and validation set losses
  8. Generate sample text for visual inspection
"""
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen

"""
AdamW:
Adam optimizers are a popular choice for training deep neural networks. 
AdamW is a variant of Adam that improves the weight decay approach, 
which aims to minimize model complexity and prevent overfitting by pernalizing larger weights.
"""

torch.manual_seed(123)

file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else
                      #"mps" if torch.backends.mps.is_available() else
                      "cpu")
model = GPTModel(GPT_CONFIG_124M).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
train_losses, val_losses, track_tokens_seen = train_model_simple(model, train_loader, val_loader,
                                                                 optimizer, device, num_epochs=10,
                                                                 eval_freq=5, eval_iter=5,
                                                                 start_context="Every effort moves you",
                                                                 tokenizer=tokenizer)
