import json
import random
import argparse

from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data/alpaca_data/alpaca_data.json')
    parser.add_argument("--save_path", type=str, default='alpaca_random_stack.json')
    parser.add_argument("--model_name_or_path", type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    print(args)

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,cache_dir='../cache')

    def get_token_count(text):
        # Tokenize the text and return the number of tokens
        # input_ids = tokenizer.encode(text, return_tensors="pt")
        return len(tokenizer.tokenize(text))

    # Load the original data from the file
    with open(args.data_path, 'r') as file:
        original_data = json.load(file)

    # Set 3 different seeds and shuffle the data 3 times, appending each shuffle to a master list
    seeds = [42, 123, 789]
    all_shuffled_data = []  # Master list to hold all shuffled data

    for seed in seeds:
        random.seed(seed)
        shuffled_data = original_data.copy()  # Create a copy of the original data for each shuffle
        random.shuffle(shuffled_data)

        i = 0
        while i < len(shuffled_data):
            # Determine random chunk size from 1 to 10
            chunk_size = random.randint(1, 10)
            end_i = i+chunk_size if i+chunk_size < len(shuffled_data) else len(shuffled_data)
            selected_items = shuffled_data[i:end_i]

            instructions_inputs = ""
            outputs = ""

            for item in selected_items:
                # Check token count before adding new item
                new_instructions_inputs = instructions_inputs + item['instruction'] + ("\n" + item['input'] if item['input'].strip() else "") + "\n\n"
                if get_token_count(new_instructions_inputs) < args.max_length:
                    instructions_inputs = new_instructions_inputs
                    outputs += (item['output'] if item['output'].strip() else " ") + "\n\n"
                else:
                    break  # Stop adding items to this chunk if token limit is exceeded

            concatenated_group = {
                "instruction": instructions_inputs.strip(),
                "output": outputs.strip(),
                "count": len(selected_items),
            }
            all_shuffled_data.append(concatenated_group)

            i += len(selected_items)  # Move to the next chunk starting position

    # Store all concatenated groups in a new JSON file
    with open(args.save_path, 'w') as outfile:
        json.dump(all_shuffled_data, outfile, indent=4)


if __name__ == "__main__":
    main()