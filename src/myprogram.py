#!/usr/bin/env python
import os
import string
import random
import csv
from collections import defaultdict
import pickle
import re
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class LanguageModel:
    """Individual language model"""
    
    def __init__(self, language):
        self.language = language
        self.char_freq = defaultdict(lambda: defaultdict(int))
        self.context_length = 6  # Use 6 previous chars for context
        
    def train(self, texts):
        """Train model on language-specific texts"""
        for text in texts:
            for i in range(1, len(text)):
                context = text[max(0, i-self.context_length):i]
                next_char = text[i]
                self.char_freq[context][next_char] += 1
                
    def get_top_chars(self, context, n=3):
        """Get top n characters given context - with fixed space handling"""
        # Check for space at end of context
        ends_with_space = bool(context and context[-1] == ' ')
        
        # Set flag for get_defaults to use
        self._after_space = ends_with_space
        
        # If context ends with space, use special handling
        if ends_with_space:
            # Direct return of space-specific defaults
            defaults = self.get_defaults()  # Will use space_defaults via flag
            return defaults[:n]
        
        # Normal processing for non-space contexts
        # Direct lookup with the right context length
        if len(context) > self.context_length:
            context_key = context[-self.context_length:]
        else:
            context_key = context
            
        freq_dict = self.char_freq[context_key]
        
        # Initialize result and seen set
        result = ''
        seen = set()
        
        # Early return path for common case
        if freq_dict:
            # Sort by frequency
            sorted_chars = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Fast path for most common case (more than n unique chars)
            char_count = 0
            for char, _ in sorted_chars:
                if char not in seen:
                    result += char
                    seen.add(char)
                    char_count += 1
                    if char_count == n:
                        return result
        
        # If we don't have enough characters or no data for context,
        # use the backoff strategy, but optimize it
        if not freq_dict or len(result) < n:
            # Try shorter context windows
            backup_length = len(context) - 1 if len(context) > 1 else 0
            while backup_length > 0:
                shorter_context = context[-backup_length:]
                freq_dict = self.char_freq[shorter_context]
                if freq_dict:
                    # Add characters from this shorter context
                    sorted_chars = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
                    for char, _ in sorted_chars:
                        if char not in seen:
                            result += char
                            seen.add(char)
                            if len(result) == n:
                                return result
                backup_length -= 1
        
        # If we still don't have enough, use defaults
        if len(result) < n:
            defaults = self.get_defaults()
            for char in defaults:
                if char not in seen and len(result) < n:
                    result += char
                    seen.add(char)
        
        return result
    
    def get_defaults(self):
        """Return default characters for this language"""
        defaults = {
            'chinese': '的一是',
            'hindi': 'कीहम',
            'russian': 'вон',
            'german': 'ein',
            'french': 'est',
            'italian': 'che',
            'spanish': 'los',
            'hebrew': 'אבג',
            'arabic': 'بتث',
            'english': 'eai',
            'latin': 'eai'  # Add explicit latin defaults
        }
        
        # For spaces, we want consonants/common first characters
        space_defaults = {
            'chinese': '我你他',
            'hindi': 'मैंवह',
            'russian': 'ятв',
            'german': 'iwd',
            'french': 'jlm',
            'italian': 'ild',
            'spanish': 'eld',
            'hebrew': 'אני',
            'arabic': 'انا',
            'english': 'itw',
            'latin': 'itw',  # Common words after space: I, the, we, etc.
            'cyrillic': 'ятв',
            'devanagari': 'मैंवह'
        }
        
        # If context ends with space, return space defaults
        if hasattr(self, '_after_space') and self._after_space:
            return space_defaults.get(self.language, 'itw')
        else:
            return defaults.get(self.language, 'eai')


class MyModel:
    """
    Multi-language model that manages alphabet-based models
    """

    def __init__(self):
        # Define language detection ranges
        self.language_ranges = {
            'chinese': (0x4e00, 0x9fff),
            'hindi': (0x0900, 0x097F),
            'russian': (0x0400, 0x04FF),
            'hebrew': (0x0590, 0x05FF),
            'arabic': (0x0600, 0x06FF)
        }
        
        # Special character sets for European languages
        self.special_chars = {
            'german': set('äöüßÄÖÜ'),
            'french': set('éèêëàâäôöùûüÿçœæ'),
            'italian': set('àèéìíîòóùú'),
            'spanish': set('áéíóúüñ¿¡')
        }
        
        # Map languages to alphabet groups
        self.language_to_alphabet = {
            'english': 'latin',
            'german': 'latin',
            'french': 'latin', 
            'italian': 'latin',
            'spanish': 'latin',
            'russian': 'cyrillic',
            'chinese': 'chinese',
            'hindi': 'devanagari',
            'hebrew': 'hebrew',
            'arabic': 'arabic'
        }
        
        # Initialize alphabet models
        self.models = {}
        self.context_length = 6

    @classmethod
    def load_language_data(cls, language):
        """Load training data for a specific language"""
        data = []
        
        # Define file paths and character sets
        if language == 'english':
            try:
                with open('src/data/english_ted_talks.csv', 'r', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    next(csv_reader)  # Skip header
                    for row in csv_reader:
                        if row:  # Make sure row isn't empty
                            text = row[0]  # Assuming transcript is first column
                            # Clean text to keep only letters and basic punctuation
                            cleaned = ''.join(c.lower() for c in text if c.isalpha() or c in ' .,!?')
                            data.append(cleaned)
            except FileNotFoundError:
                print("English TED talks file not found.")
        elif language == 'german':
            try:
                with open('src/data/german_text.txt', 'r', encoding='utf-8') as f:
                    for line in f:
                        # Clean text to keep only letters and basic punctuation
                        cleaned = ''.join(c.lower() for c in line if c.isalpha() or c in ' .,!?' or c in 'äöüßÄÖÜ')
                        data.append(cleaned)
            except FileNotFoundError:
                print("German text file not found.")
        elif language == 'russian':
            try:
                with open('src/data/russian_text.txt', 'r', encoding='utf-8') as f:
                    for line in f:
                        # Clean text to keep only letters and basic punctuation
                        cleaned = ''.join(c.lower() for c in line if c.isalpha() or c in ' .,!?' or ('\u0400' <= c <= '\u04FF'))
                        data.append(cleaned)
            except FileNotFoundError:
                print("Russian text file not found.")
        elif language == 'chinese':
            try:
                with open('src/data/chinese_text.txt', 'r', encoding='utf-8') as f:
                    for line in f:
                        cleaned_line = line.strip()
                        if cleaned_line:  # Skip empty lines
                            data.append(cleaned_line)
            except FileNotFoundError:
                print("Chinese text file not found.")
        elif language == 'hindi':
            try:
                with open('src/data/hindi_text.txt', 'r', encoding='utf-8') as f:
                    for line in f:
                        # Clean text to keep only letters and basic punctuation
                        cleaned = ''.join(c.lower() for c in line if c.isalpha() or c in ' .,!?' or ('\u0900' <= c <= '\u097F'))
                        data.append(cleaned)
            except FileNotFoundError:
                print("Hindi text file not found.")
        elif language == 'french':
            try:
                with open('src/data/french_text.txt', 'r', encoding='utf-8') as f:
                    for line in f:
                        # Clean text to keep only letters and basic punctuation
                        cleaned = ''.join(c.lower() for c in line if c.isalpha() or c in ' .,!?' or c in 'éèêëàâäôöùûüÿçœæ')
                        data.append(cleaned)
            except FileNotFoundError:
                print("French text file not found.")
        elif language == 'italian':
            try:
                with open('src/data/italian_text.txt', 'r', encoding='utf-8') as f:
                    for line in f:
                        # Clean text to keep only letters and basic punctuation
                        cleaned = ''.join(c.lower() for c in line if c.isalpha() or c in ' .,!?' or c in 'àèéìíîòóùú')
                        data.append(cleaned)
            except FileNotFoundError:
                print("Italian text file not found.")
        elif language == 'spanish':
            try:
                with open('src/data/spanish_text.txt', 'r', encoding='utf-8') as f:
                    for line in f:
                        # Clean text to keep only letters and basic punctuation
                        cleaned = ''.join(c.lower() for c in line if c.isalpha() or c in ' .,!?' or c in 'áéíóúüñ¿¡')
                        data.append(cleaned)
            except FileNotFoundError:
                print("Spanish text file not found.")
        elif language == 'hebrew':
            try:
                with open('src/data/hebrew_text.txt', 'r', encoding='utf-8') as f:
                    for line in f:
                        # Clean text to keep only letters and basic punctuation
                        cleaned = ''.join(c.lower() for c in line if c.isalpha() or c in ' .,!?' or (0x0590 <= ord(c) <= 0x05FF))
                        data.append(cleaned)
            except FileNotFoundError:
                print("Hebrew text file not found.")
        elif language == 'arabic':
            try:
                with open('src/data/arabic_text.txt', 'r', encoding='utf-8') as f:
                    for line in f:
                        # Clean text to keep only letters and basic punctuation
                        cleaned = ''.join(c.lower() for c in line if c.isalpha() or c in ' .,!?' or (0x0600 <= ord(c) <= 0x06FF))
                        data.append(cleaned)
            except FileNotFoundError:
                print("Arabic text file not found.")

        return data

    @classmethod
    def load_training_data(cls):
        """For backwards compatibility"""
        # This method loads all data together
        all_data = []
        languages = ['english', 'german', 'russian', 'chinese', 'hindi', 'french', 
                    'italian', 'spanish', 'hebrew', 'arabic']
        languages_with_latin = ['latin', 'russian', 'chinese', 'hindi', 
                     'hebrew', 'arabic']
        
        for language in languages:
            all_data.extend(cls.load_language_data(language))
            
        return all_data

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

    def detect_alphabet(self, text):
        """Detect the alphabet of a text string - optimized version"""
        # Quick check for empty text
        if not text:
            return 'latin'
            
        # Check first a few characters instead of the entire text
        sample = text[-min(10, len(text)):]
        
        # Use a more efficient approach with any() and fewer function calls
        for char in sample:
            code_point = ord(char)
            if 0x4e00 <= code_point <= 0x9fff:
                return 'chinese'
            if 0x0900 <= code_point <= 0x097F:
                return 'devanagari'
            if 0x0400 <= code_point <= 0x04FF:
                return 'cyrillic'
            if 0x0590 <= code_point <= 0x05FF:
                return 'hebrew'
            if 0x0600 <= code_point <= 0x06FF:
                return 'arabic'
        
        # Check for Latin with special characters - only if needed
        for char in sample:
            if char in 'äöüßÄÖÜéèêëàâôöùûüÿçœæàèéìíîòóùúáéíóúüñ¿¡':
                return 'latin'
        
        # Default to Latin
        return 'latin'

    def get_top_chars(self, context, n=3):
        """Get predictions using the appropriate alphabet model"""
        # Use cache to avoid redetecting alphabets
        if not hasattr(self, '_alphabet_cache'):
            self._alphabet_cache = {}
        
        # Check if we've already detected this context
        if context in self._alphabet_cache:
            alphabet = self._alphabet_cache[context]
        else:
            # Detect alphabet and cache it
            alphabet = self.detect_alphabet(context)
            self._alphabet_cache[context] = alphabet
            
            # Limit cache size
            if len(self._alphabet_cache) > 10000:
                self._alphabet_cache = {k: self._alphabet_cache[k] 
                                       for k in list(self._alphabet_cache.keys())[-5000:]}
        
        # Handle space-ending contexts directly
        if context and context[-1] == ' ':
            if alphabet in self.models:
                # Set space flag on the model
                self.models[alphabet]._after_space = True
                result = self.models[alphabet].get_top_chars(context, n)
                self.models[alphabet]._after_space = False  # Reset flag
                return result
            elif 'latin' in self.models:
                # Fallback for unknown alphabets
                self.models['latin']._after_space = True
                result = self.models['latin'].get_top_chars(context, n)
                self.models['latin']._after_space = False  # Reset flag
                return result
            else:
                return 'itw'  # Common first letters after space
        
        # Normal processing for non-space contexts
        if alphabet in self.models:
            return self.models[alphabet].get_top_chars(context, n)
            
        # Fallback to Latin if no model found
        if 'latin' in self.models:
            return self.models['latin'].get_top_chars(context, n)
            
        # Ultimate fallback
        return 'eai'

    def run_train(self, data, work_dir):
        """Train separate models for each alphabet"""
        print("Training alphabet-based models...")
        
        # Define alphabet groups and their associated languages
        alphabet_groups = {
            'latin': ['english', 'german', 'french', 'italian', 'spanish'],
            'cyrillic': ['russian'],
            'chinese': ['chinese'],
            'devanagari': ['hindi'],
            'hebrew': ['hebrew'],
            'arabic': ['arabic']
        }
        
        # Create and train each alphabet model
        for alphabet, languages in alphabet_groups.items():
            print(f"Creating {alphabet} alphabet model...")
            
            # Create a model for this alphabet
            self.models[alphabet] = LanguageModel(alphabet)
            
            # Collect all text data for languages in this alphabet group
            alphabet_data = []
            for language in languages:
                language_data = self.load_language_data(language)
                if language_data:
                    alphabet_data.extend(language_data)
            
            # Train the alphabet model if we have data
            if alphabet_data:
                self.models[alphabet].train(alphabet_data)
            else:
                print(f"No data found for {alphabet}, skipping model creation")

    def run_pred(self, data):
        """Optimized prediction function"""
        preds = []
        
        # Pre-initialize some variables
        context_length = self.context_length
        
        for inp in data:
            # Get context based on length - avoid function calls
            if len(inp) >= context_length:
                context = inp[-context_length:]
            else:
                context = inp
            
            # Get top 3 characters using alphabet-specific model
            top_3 = self.get_top_chars(context)                
            preds.append(top_3)
            
        return preds

    def save(self, work_dir):
        """Save each alphabet model separately"""
        os.makedirs(work_dir, exist_ok=True)
        
        # Save alphabet models
        for alphabet, model in self.models.items():
            model_dir = os.path.join(work_dir, alphabet)
            os.makedirs(model_dir, exist_ok=True)
            
            # Convert defaultdict to regular dict for serialization
            char_freq_dict = dict(model.char_freq)
            for context in char_freq_dict:
                char_freq_dict[context] = dict(char_freq_dict[context])
            
            with open(os.path.join(model_dir, 'model.checkpoint'), 'wb') as f:
                pickle.dump(char_freq_dict, f)
                
        # Save model metadata
        with open(os.path.join(work_dir, 'alphabets.txt'), 'w', encoding='utf-8') as f:
            for alphabet in self.models:
                f.write(f"{alphabet}\n")

    @classmethod
    def load(cls, work_dir):
        model = cls()
        
        # Load available alphabet models
        alphabets_file = os.path.join(work_dir, 'alphabets.txt')
        if os.path.exists(alphabets_file):
            with open(alphabets_file, 'r', encoding='utf-8') as f:
                alphabets = [line.strip() for line in f if line.strip()]
                
            for alphabet in alphabets:
                model_dir = os.path.join(work_dir, alphabet)
                checkpoint_path = os.path.join(model_dir, 'model.checkpoint')
                
                if os.path.exists(checkpoint_path):
                    # Create alphabet model
                    alpha_model = LanguageModel(alphabet)
                    
                    # Load frequency data
                    with open(checkpoint_path, 'rb') as f:
                        char_freq_dict = pickle.load(f)
                        # Convert to defaultdict
                        alpha_model.char_freq = defaultdict(lambda: defaultdict(int))
                        for context, freq in char_freq_dict.items():
                            alpha_model.char_freq[context].update(freq)
                    
                    # Add to models dictionary
                    model.models[alphabet] = alpha_model
        else:
            # Try loading language models for backward compatibility
            languages_file = os.path.join(work_dir, 'languages.txt')
            if os.path.exists(languages_file):
                print("Found language models instead of alphabet models, converting...")
                with open(languages_file, 'r', encoding='utf-8') as f:
                    languages = [line.strip() for line in f if line.strip()]
                
                # Group languages by alphabet
                alphabet_models = {}
                for language in languages:
                    model_dir = os.path.join(work_dir, language)
                    checkpoint_path = os.path.join(model_dir, 'model.checkpoint')
                    
                    if os.path.exists(checkpoint_path):
                        # Determine which alphabet this language belongs to
                        alphabet = language
                        if language in ('english', 'german', 'french', 'italian', 'spanish'):
                            alphabet = 'latin'
                        elif language == 'russian':
                            alphabet = 'cyrillic'
                        elif language == 'hindi':
                            alphabet = 'devanagari'
                        
                        # Create the alphabet model if it doesn't exist
                        if alphabet not in alphabet_models:
                            alphabet_models[alphabet] = defaultdict(lambda: defaultdict(int))
                        
                        # Load and merge the language model data
                        with open(checkpoint_path, 'rb') as f:
                            char_freq_dict = pickle.load(f)
                            for context, freq in char_freq_dict.items():
                                for char, count in freq.items():
                                    alphabet_models[alphabet][context][char] += count
                
                # Convert the merged data into models
                for alphabet, freq_dict in alphabet_models.items():
                    alpha_model = LanguageModel(alphabet)
                    alpha_model.char_freq = defaultdict(lambda: defaultdict(int))
                    for context, freq in freq_dict.items():
                        alpha_model.char_freq[context].update(freq)
                    model.models[alphabet] = alpha_model
                    print(f"Converted and loaded {alphabet} model")
            else:
                # For backwards compatibility with the very old format
                checkpoint_path = os.path.join(work_dir, 'model.checkpoint')
                if os.path.exists(checkpoint_path):
                    # Create a unified Latin model (legacy format)
                    unified_model = LanguageModel('latin')
                    
                    with open(checkpoint_path, 'rb') as f:
                        char_freq_dict = pickle.load(f)
                        unified_model.char_freq = defaultdict(lambda: defaultdict(int))
                        for context, freq in char_freq_dict.items():
                            unified_model.char_freq[context].update(freq)
                    
                    model.models['latin'] = unified_model
                    print("Loaded legacy unified model as Latin alphabet model")
                
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
        print('Instantiating model')
        model = MyModel()
        print('Training')
        model.run_train(None, args.work_dir)
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