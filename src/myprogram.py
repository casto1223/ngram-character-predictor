#!/usr/bin/env python
import os
import string
import random
import csv
from collections import defaultdict
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self):
        self.char_freq = defaultdict(lambda: defaultdict(int))
        self.context_length = 6  # Use 6 previous chars for context
        # self.print_chars_repeatedly("")
        # self.clean_text('src/data/hin_news_2022_10K-sentences.txt', 'src/data/hindi_text.txt')

    @classmethod
    def load_training_data(cls):
        data = []
        file_paths = [
            ('src/data/english_ted_talks.csv', 'English TED talks', True),
            ('src/data/german_text.txt', 'German text', False),
            ('src/data/russian_text.txt', 'Russian text', False),
            ('src/data/chinese_text.txt', 'Chinese text', False),
            ('src/data/hindi_text.txt', 'Hindi text', False)
        ]

        # Define character validation functions outside the loops
        def is_valid_char(c, file_type):
            if c.isalpha() or c in ' .,!?':
                return True
            if file_type in ['Russian text', 'Hindi text'] and '\u0400' <= c <= '\u04FF':
                return True
            if file_type == 'Hindi text' and '\u0900' <= c <= '\u097F':
                return True
            return False

        for file_path, file_type, is_csv in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if is_csv:
                    # Process CSV file (English TED talks)
                        csv_reader = csv.reader(f)
                        next(csv_reader)  # Skip header
                        for row in csv_reader:
                            if row:
                                text = row[0]
                                cleaned = ''.join(c.lower() for c in text if is_valid_char(c, file_type))
                                data.append(cleaned)
                    elif file_type == 'Chinese text':
                    # Process Chinese text with minimal cleaning
                        for line in f:
                            cleaned_line = line.strip()
                            if cleaned_line:
                                data.append(cleaned_line)
                    else:
                    # Process other text files
                        for line in f:
                            cleaned = ''.join(c.lower() for c in line if is_valid_char(c, file_type))
                            data.append(cleaned)
            except FileNotFoundError:
                print(f"{file_type} file not found. Continuing with other data sources.")
        
        return data

    @classmethod
    def load_test_data(cls, fname):
        data = []
        try:
            with open(fname, 'r', encoding='utf-8') as f:
                for line in f:
                    inp = line.strip()
                    if inp:  # Skip empty lines
                        data.append(inp)
        except Exception as e:
            print(f"Error loading test data: {e}")
            raise
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        try:
            with open(fname, 'w', encoding='utf-8') as f:
                for p in preds:
                    f.write(f"{p}\n")
        except Exception as e:
            print(f"Error writing predictions: {e}")
            raise

    def run_train(self, data, work_dir):
        print("Training on data...")
        for text in data:
            # Process text in appropriate chunks
            for i in range(1, len(text)):  # Start at 1 since we're predicting the next character
                # Get context (previous characters)
                context = text[max(0, i-self.context_length):i]
                # Character to predict
                next_char = text[i]
                # Update frequency dictionary
                self.char_freq[context][next_char] += 1
            

    def get_top_chars(self, context, n=3):
        # Get frequency dict for this context
        freq_dict = self.char_freq[context[-self.context_length:]]
        
        # Debug context analysis
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in context)
        has_hindi = any('\u0900' <= char <= '\u097F' for char in context)
        has_russian = any('\u0400' <= char <= '\u04FF' for char in context)
        context_type = "Chinese" if has_chinese else "Hindi" if has_hindi else "Russian" if has_russian else "English"
        
        # If no data for this context, back off to shorter context
        original_context = context
        backup_length = len(context)
        while not freq_dict and backup_length > 0:
            backup_length -= 1
            shorter_context = context[-backup_length:] if backup_length > 0 else ""
            freq_dict = self.char_freq[shorter_context]
        
        # If still no data, use appropriate defaults based on input language
        if not freq_dict:
            if has_chinese:
                return '的一是'  # Common Chinese characters
            elif has_hindi:
                return 'कीहम'  # Common Hindi characters
            elif has_russian:
                return 'вон'  # Common Russian characters
            else:
                return 'eai'  # Default for English
        
        # Sort by frequency and return top n unique characters
        sorted_chars = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
        result = ''
        seen = set()
        
        for char, freq in sorted_chars:
            if char not in seen:
                result += char
                seen.add(char)
                if len(result) == n:
                    break
        
        # If we don't have enough unique characters, pad with defaults
        defaults = '的一是' if has_chinese else 'कीहम' if has_hindi else 'вон' if has_russian else 'eai'
        i = 0
        while len(result) < n:
            if defaults[i] not in seen:
                result += defaults[i]
                seen.add(defaults[i])
            i = (i + 1) % len(defaults)
        
        return result

    def run_pred(self, data):
        preds = []
        for inp in data:
            # Get last n characters as context
            inp = inp.lower()
            context = inp[-self.context_length:] if len(inp) >= self.context_length else inp
            top_3 = self.get_top_chars(context)
            preds.append(top_3)
        return preds

    def save(self, work_dir):
        checkpoint_path = os.path.join(work_dir, 'model.checkpoint')
        # Convert defaultdict to regular dict for serialization
        char_freq_dict = dict(self.char_freq)
        for context in char_freq_dict:
            char_freq_dict[context] = dict(char_freq_dict[context])
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(char_freq_dict, f)

    @classmethod
    def load(cls, work_dir):
        model = cls()
        checkpoint_path = os.path.join(work_dir, 'model.checkpoint')
        with open(checkpoint_path, 'rb') as f:
            char_freq_dict = pickle.load(f)
            # Convert back to defaultdict
            model.char_freq = defaultdict(lambda: defaultdict(int))
            for context, freq in char_freq_dict.items():
                model.char_freq[context].update(freq)
        return model

    @classmethod
    def print_chars_repeatedly(self, s):
        # Print progressive expansion of the string
        for j in range(1, len(s) + 1):
            print(s[:j])
        
        # Print each character on a new line
        for char in s:
            print(char)

    @classmethod
    def clean_text(cls, input_file, output_file):
        """
        Clean German text data by removing numbers and leading spaces at the start of each line
        
        Args:
            input_file (str): Path to the input file
            output_file (str): Path to the output file where cleaned text will be saved
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    # Use regex-like approach to remove number prefix and spaces
                    cleaned_line = line.strip()
                    # Find position after the initial number and spaces
                    pos = 0
                    # Skip initial numbers
                    while pos < len(cleaned_line) and cleaned_line[pos].isdigit():
                        pos += 1
                    # Skip spaces after numbers
                    while pos < len(cleaned_line) and cleaned_line[pos].isspace():
                        pos += 1
                    # Write the cleaned text
                    if pos < len(cleaned_line):
                        f_out.write(cleaned_line[pos:] + '\n')
            
            print(f"Cleaned text saved to {output_file}")
        except Exception as e:
            print(f"Error cleaning text: {e}")


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
