from importlib.metadata import version
import tiktoken

print("tiktoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)

# Try the BPE tokenizer from the tiktoken library on the unknown words “Akwirw ier” and print the individual token IDs.
# Then, call the decode function on each of the resulting integers in this list to reproduce the mapping.
# Lastly, call the decode method on the token IDs to check whether it can reconstruct the original input, “Akwirw ier.”
akwirwier = "Akwirw ier"
akwirwier_integers = tokenizer.encode(akwirwier, allowed_special={"<|endoftext|>"})
print({tokenizer.decode_single_token_bytes(i):i for i in akwirwier_integers})
print(tokenizer.decode(akwirwier_integers))
