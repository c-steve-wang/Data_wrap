import argparse
import json
from transformers import AutoTokenizer

def concatenate_json(input_path, output_path, model_name_or_path, max_length):
    # Initialize the tokenizer with the provided model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir='../cache')
    
    def get_token_count(text):
        # Tokenize the text and return the number of tokens
        return len(tokenizer.tokenize(text))

    with open(input_path, 'r') as f:
        data = json.load(f)
    
    concatenated_data = []

    for item in data:
        instruction = item['instruction']
        input_ = item['input']
        output = item['output']
        
        # Concatenate instruction and input if input exists, then check token count
        concatenated_instruction_input = instruction + ("\n" + input_ if input_.strip() else "") 
        outputs = (item['output'] if item['output'].strip() else " ") 
        if get_token_count(concatenated_instruction_input) < max_length:
            # If token count is within the limit, add the concatenated data to the list
            concatenated_data.append({
                "instruction": concatenated_instruction_input.strip(),
                "output": outputs.strip(),
            })
        else:
            # Handle the case where the concatenated instruction and input exceed the max token count
            print(f"Warning: Skipping an item because it exceeds the max_length of {max_length} tokens.")

    # Write the concatenated data to the output file
    with open(output_path, 'w') as f:
        json.dump(concatenated_data, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Concatenate instruction and input for each item in JSON using a specified tokenizer.')
    parser.add_argument('--input_path', type=str, default='./alpaca_data.json', help='Path to the input JSON file')
    parser.add_argument('--output_path', type=str, default='./concatenated_data.json', help='Path to the output JSON file')
    parser.add_argument('--max_length', type=int, default=2048, help='Maximum length of concatenated instruction and input')
    parser.add_argument("--model_name_or_path", type=str, default='meta-llama/Llama-2-7b-hf', help='Model name or path for tokenizer')
    args = parser.parse_args()

    concatenate_json(args.input_path, args.output_path, args.model_name_or_path, args.max_length)

if __name__ == "__main__":
    main()
