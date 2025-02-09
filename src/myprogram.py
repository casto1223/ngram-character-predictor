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

    @classmethod
    def load_training_data(cls):
        # your code here
        # this particular model doesn't train
        data = []
        with open('src/data/english_ted_talks.csv', 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # Skip header
            for row in csv_reader:
                if row:  # Make sure row isn't empty
                    text = row[0]  # Assuming transcript is first column
                    # Clean text to keep only letters and basic punctuation
                    cleaned = ''.join(c.lower() for c in text if c.isalpha() or c in ' .,!?')
                    data.append(cleaned)
        return data

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # your code here
        for text in data:
            for i in range(len(text)-1):
                context = text[max(0, i-self.context_length):i]
                next_char = text[i]
                self.char_freq[context][next_char] += 1

    def get_top_chars(self, context, n=3):
        # Get frequency dict for this context
        freq_dict = self.char_freq[context[-self.context_length:]]
        
        # If no data for this context, back off to shorter context
        while not freq_dict and len(context) > 0:
            context = context[1:]
            freq_dict = self.char_freq[context]
        
        # If still no data, use default letters
        if not freq_dict:
            return 'eai'
        
        # Sort by frequency and return top n unique characters
        sorted_chars = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
        result = ''
        seen = set()
        for char, _ in sorted_chars:
            if char not in seen:
                result += char
                seen.add(char)
                if len(result) == n:
                    break
        
        # If we don't have enough unique characters, pad with defaults
        defaults = 'eai'
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
